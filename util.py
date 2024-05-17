import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from metric import cal_ncc
PI = math.pi
cal_mse = nn.MSELoss()

# Convert numpy to tensor
def tensor_exp2torch(T, BATCH_SIZE, device):
    T = np.expand_dims(T, axis=0)
    T = np.expand_dims(T, axis=0)
    T = np.repeat(T, BATCH_SIZE, axis=0)

    T = torch.tensor(T, dtype=torch.float, requires_grad=True, device=device)

    return T


# Count network parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
Defines ProST canonical geometries
input:
    CT_PATH: file path of CT segmentation
       flag: downsample factor
     proj_x: project size
     ISFlip: True if Z(IS) is flipped
output:
           param: src, det, pix_spacing, step_size, det_size
          ct_vol: volume used for training DeepNet, we use CT segmentation
    ray_proj_mov: detector plane variable
       corner_pt: 8 corner points of input volume
     norm_factor: translation normalization factor
'''


def input_param(CT_PATH, BATCH_SIZE, flag=1, proj_x=1024, ISFlip=False, device='cuda'):
    ct_vol = sitk.ReadImage(CT_PATH)
    vol_spacing = ct_vol.GetSpacing()[1]
    ct_vol = sitk.GetArrayFromImage(ct_vol)
    ct_vol = ct_vol.transpose((2, 1, 0))
    N = ct_vol.shape[0]
    pixel_id_detect = 0.19959
    src_det = 5069.9 * pixel_id_detect
    iso_center = src_det - N / 2 * pixel_id_detect
    det_size = proj_x
    pix_spacing = pixel_id_detect * 1024 / det_size
    step_size = 2
    vol_size = 512 / flag

    norm_factor = (vol_size * vol_spacing / 2)
    src = (src_det - iso_center) / norm_factor
    det = -iso_center / norm_factor
    pix_spacing = pix_spacing / norm_factor
    step_size = step_size / norm_factor
    param = [src, det, pix_spacing, step_size, det_size]

    ct_vol = tensor_exp2torch(ct_vol, BATCH_SIZE, device)
    corner_pt = create_cornerpt(BATCH_SIZE, device)
    ray_proj_mov = np.zeros((det_size, det_size))
    ray_proj_mov = tensor_exp2torch(ray_proj_mov, BATCH_SIZE, device)
    return param, det_size, ct_vol, ray_proj_mov, corner_pt, norm_factor



def pose2init_param(rtvec_param, BATCH_SIZE):
    rz, rx, ry, tx, ty, tz = rtvec_param
    init_param = np.repeat([[rz, ry, rx, -tz, -ty, tx]], BATCH_SIZE, 0)
    return init_param


def param2rtvec(param, norm_factor, device):
    param[:, :3] = param[:, :3] / 180 * PI
    param[:, 3:] = param[:, 3:] / norm_factor
    rtvec = torch.tensor(param, dtype=torch.float, requires_grad=True, device=device)
    return rtvec


def distribute_func(BATCH_SIZE, distribution = 'U'):
    if distribution == 'U':
        result = np.random.uniform(-1, 1, (BATCH_SIZE, 6))
    elif distribution == 'N':
        result = np.random.uniform(0, 1, (BATCH_SIZE, 6))
    return result

# Generate initial rotation-translation vectors and transform matrices for target and initial parameters
def init_rtvec(BATCH_SIZE, device, norm_factor, center = [90, 0, 0, 700, 0, 0], distribution = 'N', manual = False, rtvec_gt_param = None, lateral = False, rtvec_gt_param_lateral = None, manual_param_range= None, iterative = False):
   
    if manual_param_range == None:
        if iterative:
            param_range = [20, 20, 20, 100, 50, 50]
        # param_range =[10,10,10,15,15,15]
        else:
            param_range = [40, 40, 40, 200, 75, 75]
    else:
        param_range=manual_param_range
    scale = pose2init_param(param_range, BATCH_SIZE)
    # Uniform Distribution/Normal distribution
    if manual:
        target = pose2init_param(rtvec_gt_param, BATCH_SIZE)
        target_param = target
        initial_param = distribute_func(BATCH_SIZE, distribution) * scale + target_param
        if lateral:
            target_lateral = pose2init_param(rtvec_gt_param_lateral, BATCH_SIZE)
            target_param_lateral = target_lateral
            initial_param_lateral = distribute_func(BATCH_SIZE, distribution) * scale + target_param_lateral
    else:
        target = pose2init_param(center, BATCH_SIZE)
        target_param = distribute_func(BATCH_SIZE, distribution) * scale + target
        if iterative:
            initial_param = distribute_func(BATCH_SIZE, distribution) * scale + target
        else:
            initial_param = distribute_func(BATCH_SIZE, distribution) * scale + target_param
        if lateral:
            target_lateral = pose2init_param([center[0] + 90].append([i for i in center[1:]]), BATCH_SIZE)
            target_param_lateral = distribute_func(BATCH_SIZE, distribution) * scale + target_lateral
            if iterative:
                initial_param_lateral = distribute_func(BATCH_SIZE, distribution) * scale + target_lateral
            else:
                initial_param_lateral = distribute_func(BATCH_SIZE, distribution) * scale + target_param_lateral
    rtvec = param2rtvec(initial_param, norm_factor, device)
    rtvec_gt = param2rtvec(target_param, norm_factor, device)
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
    transform_mat3x4_gt = set_matrix(BATCH_SIZE, device, rtvec_gt)
    if lateral:
        rtvec_lateral = param2rtvec(initial_param_lateral, norm_factor, device)
        rtvec_gt_lateral = param2rtvec(target_param_lateral, norm_factor, device)
        transform_mat3x4_lateral = set_matrix(BATCH_SIZE, device, rtvec_lateral)
        transform_mat3x4_gt_lateral = set_matrix(BATCH_SIZE, device, rtvec_gt_lateral)
        return transform_mat3x4, transform_mat3x4_gt, rtvec, rtvec_gt, \
                transform_mat3x4_lateral, transform_mat3x4_gt_lateral, rtvec_lateral, rtvec_gt_lateral
    else:
        return transform_mat3x4, transform_mat3x4_gt, rtvec, rtvec_gt


# Create corner points
def create_cornerpt(BATCH_SIZE, device):
    corner_pt = np.array(
        [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    corner_pt = torch.tensor(corner_pt.astype(float), requires_grad=False).type(torch.FloatTensor)
    corner_pt = corner_pt.unsqueeze(0).to(device)
    corner_pt = corner_pt.repeat(BATCH_SIZE, 1, 1)

    return corner_pt


# Repeat tensor
def _repeat(x, n_repeats):
    with torch.no_grad():
        rep = torch.ones((1, n_repeats), dtype=torch.float32).cuda()

    return torch.matmul(x.view(-1, 1), rep).view(-1)



# Calculate mean of TRE
def cal_mTRE(ct_vol, rtvec_gt_param, rtvec_param, BATCH_SIZE, device):
    '''
    A widely used 3-D error measure is the target registration error (TRE), where the “targets” in the TRE calculation can
    be predefined locations (either fiducials or landmarks), surface points, or arbitrary chosen points inside a region of interest.
    '''
    return mTRE




def set_matrix(BATCH_SIZE, device, proj_parameters):
    '''
    Convert rotation-translation vector into transform matrix
    Args:
        proj_parameters: 6DoF parameters in the order of tx, tz, ty, rz, ry, rx -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    '''
    return transform_mat3x4



def pose2rtvec(pose, device, norm_factor):
    '''
    Convert pose parameters to rotation-translation vector
    Args:
        pose: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz-- [B, 6]
    Returns:
        A 6DoF parameters in the order of tx, tz, ty, rz, ry, rx -- [B, 6]
    '''
    return rtvec



def rtvec2pose(rtvec, norm_factor, device):
    '''
    Convert rotation-translation vector to pose parameters
    Args:
        rtvec: 6DoF parameters in the order of tx, tz, ty, rz, ry, rx-- [B, 6]
    Returns:
        A 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    '''
    return pose


# Convert pose parameters to transform matrix
def pose2mat(pose, BATCH_SIZE, device, norm_factor):
    rtvec = pose2rtvec(pose, device, norm_factor)
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
    return transform_mat3x4



def seed_everything(seed):
    '''
    Seed everything for random steps
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






def eval(test_func, test_epoch_num, DF_PATH,include_ini=False):
    '''
    Evaluation for testing
    Args:
        test_func: name of the function for evaluating
        test_epoch_num: number of tests -- int
        DF_path: path of the file to save the result(csv format)
        include_ini: whether to evaluate and record the initial value -- bool
    Returns:
        None
    '''
class SGDWithBounds(optim.SGD):
    def __init__(self, params, bounds, lr=0.01, momentum=0.6, dampening=0.45, weight_decay=1e-8, nesterov=False):
        super(SGDWithBounds, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        self.bounds = bounds

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param = p.data

                # Adding Boundary Constraints
                param.add_(grad, alpha=-group['lr'])
                param.clamp_(self.bounds[0], self.bounds[1])
                p.data = param
