import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import time
from module import ProST
from util import input_param, init_rtvec, cal_mTRE, rtvec2pose, eval, domain_randomization, set_matrix, seed_everything, SGDWithBounds
from metric import gradncc, ncc, ngi, nccl, PW_NCC, MPW_NCC
from metric import MultiscaleNormalizedCrossCorrelation2d as MSP_NCC
from metric import MultiscaleGradientNormalizedCrossCorrelation2d as MSP_GC
import numpy as np
import math
import warnings

seed = 772512  # 182501, 852097, 881411
print('seed:', seed)
seed_everything(seed)

warnings.filterwarnings("ignore")
np.set_printoptions(formatter={'float': '{:.4f}'.format})

DATA_SET = '../../Data/ct/256/test'
X_RAY_SET = '../../Data/x_ray/256/synthetic/test_1000'
x_ray_files = os.listdir(X_RAY_SET)
x_ray_list = []
for x_ray_file in x_ray_files:
    x_ray_list.append(x_ray_file)

BATCH_SIZE = 1
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
zFlip = False
proj_size = 256
flag = 2
initmodel = ProST().to(device)

def similarity_measure(rtvec, target, CT_vol, ray_proj_mov, corner_pt, param, metric='ncc'):
    transform_mat3x4 = set_matrix(BATCH_SIZE, device, rtvec)
    # s = time.time()
    moving = initmodel(CT_vol, ray_proj_mov, transform_mat3x4, corner_pt, param)
    # e = time.time()
    # print(f'prost: {e - s}')
    if metric=='ncc':
        return ncc(target,moving)
    elif metric=='msp_ncc':
        operator= MSP_NCC(patch_sizes=[None, 13], patch_weights=[0.5, 0.5])
        return 1 - operator(target,moving)
    elif metric=='mpw_ncc':
        return MPW_NCC(target,moving,patch_sizes=[7])
    elif metric=='msp_gc':
        operator=1 - MSP_GC(patch_sizes=[None,13],patch_weights=[0.5,0.5])
        return operator(target,moving)
    elif metric=='gc':
        return gradncc(target,moving)
    elif metric=='ngi':
        return ngi(target,moving)
    elif metric=='nccl':
        return nccl(target,moving)
    elif metric=='pw_ncc':
        return PW_NCC(target,moving)
    else:
        raise ValueError(
            f"{metric} not recongnized, must be  ['ncc', 'gc', 'ngi'...]"
        )
    
def Gradient_Descent(i, metric='msp_ncc'):
    torch.cuda.empty_cache()
    X_RAY_NAME = x_ray_list[i]
    X_RAY_SPLIT = X_RAY_NAME.split('.nii.gz')[0].split('_')
    CT_NAME = X_RAY_SPLIT[0]
    pose_gt = [float(i) for i in X_RAY_SPLIT[1:7]]
    X_RAY_PATH = f'{X_RAY_SET}/{X_RAY_NAME}'
    CT_PATH = DATA_SET + '/' + CT_NAME + '_256.nii.gz'
    param, _, CT_vol, target, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, BATCH_SIZE, flag, proj_size, X_RAY_PATH=X_RAY_PATH)
    
    _, _, rtvec_init, _ = init_rtvec(BATCH_SIZE, device, norm_factor, manual_pose_distribution='N', manual_pose_range=[15, 15, 15, 25, 25, 25], center=[90, 0, 0, 700, 0, 0], iterative=True)
    pose_init = rtvec2pose(rtvec_init, norm_factor, device).cpu().detach().numpy().squeeze()
    
    rtvec_rot1 = rtvec_init[:, :1].clone()
    rtvec_rot2 = rtvec_init[:, 1:2].clone()
    rtvec_rot3 = rtvec_init[:, 2:3].clone()
    rtvec_trans1 = rtvec_init[:, 3:4].clone()
    rtvec_trans2 = rtvec_init[:, 4:5].clone()
    rtvec_trans3 = rtvec_init[:, 5:].clone()
    rtvec_rot1.requires_grad = True
    rtvec_rot2.requires_grad = True
    rtvec_rot3.requires_grad = True
    rtvec_trans1.requires_grad = True
    rtvec_trans2.requires_grad = True
    rtvec_trans3.requires_grad = True
    # bound=[[50 * math.pi / 180, 130 * math.pi / 180],
    #        [-40 * math.pi / 180, 40 * math.pi / 180],
    #        [-40 * math.pi / 180, 40 * math.pi / 180],
    #        [-100 / norm_factor, 100 / norm_factor],
    #        [-100 / norm_factor, 100 / norm_factor],
    #        [500 / norm_factor, 900 / norm_factor]]
    # bound=[[45 * math.pi / 180, 135 * math.pi / 180],
    #        [-45 * math.pi / 180, 45 * math.pi / 180],
    #        [-45 * math.pi / 180, 45 * math.pi / 180],
    #        [-50 / norm_factor, 50 / norm_factor],
    #        [-50 / norm_factor, 50 / norm_factor],
    #        [600 / norm_factor, 800 / norm_factor]]
    bound=[[60 * math.pi / 180, 120 * math.pi / 180],
           [-35 * math.pi / 180, 30 * math.pi / 180],
           [-30 * math.pi / 180, 30 * math.pi / 180],
           [-50 / norm_factor, 50 / norm_factor],
           [-50 / norm_factor, 50 / norm_factor],
           [650 / norm_factor, 750 / norm_factor]]
    bound = np.array(bound)

    # with torch.no_grad():
    #     target = initmodel(FC_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt, param)

    # lr1 = 7.5e-3
    # lr2 = 7.5e-3
    # lr3 = 7.5e-3
    # lr4 = 7.5e-2
    # lr5 = 7.5e-2
    # lr6 = 7.5e-2
    # optimizer = torch.optim.Adam(
    #     [
    #         {'params':[rtvec_rot1], 'lr':lr1, 'betas':(0.9, 0.999)},
    #         {'params':[rtvec_rot2], 'lr':lr2, 'betas':(0.9, 0.999)},
    #         {'params':[rtvec_rot3], 'lr':lr3, 'betas':(0.9, 0.999)},
    #         {'params':[rtvec_trans1], 'lr':lr4, 'betas':(0.9, 0.999)},
    #         {'params':[rtvec_trans2], 'lr':lr5, 'betas':(0.9, 0.999)},
    #         {'params':[rtvec_trans3], 'lr':lr6, 'betas':(0.9, 0.999)},
    #     ],
    #     bounds, lr=0.01, momentum=0.6, dampening=0.45, weight_decay=1e-8
    # )
    lr_rot = 5e-2
    lr_trans = 1e-2
    lr1 = lr_rot
    lr2 = lr_rot
    lr3 = lr_rot
    lr4 = lr_trans
    lr5 = lr_trans
    lr6 = lr_trans
    optimizer1 = SGDWithBounds(params=[rtvec_rot1], bounds=bound[0], lr=lr1, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer2 = SGDWithBounds(params=[rtvec_rot2], bounds=bound[1], lr=lr2, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer3 = SGDWithBounds(params=[rtvec_rot3], bounds=bound[2], lr=lr3, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer4 = SGDWithBounds(params=[rtvec_trans1], bounds=bound[3], lr=lr4, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer5 = SGDWithBounds(params=[rtvec_trans2], bounds=bound[4], lr=lr5, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer6 = SGDWithBounds(params=[rtvec_trans3], bounds=bound[5], lr=lr6, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    # print(f'lr_rot: {lr_rot} lr_trans: {lr_trans}')
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=25, gamma=0.9)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=25, gamma=0.9)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=25, gamma=0.9)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=25, gamma=0.9)
    scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=25, gamma=0.9)
    scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=25, gamma=0.9)
    start = time.time()
    rtvec_res = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
    min_generation = 0
    min_loss = similarity_measure(torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1), target, CT_vol, ray_proj_mov, corner_pt, param, metric)

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    optimizer5.zero_grad()
    optimizer6.zero_grad()
    min_loss.backward()
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    optimizer6.step()
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()
    scheduler6.step()

    # print(f'pose_gt: {rtvec2pose(rtvec_gt, norm_factor, device).cpu().detach().numpy().squeeze()}')
    # print(f'pose_init: {rtvec2pose(rtvec_init, norm_factor, device).cpu().detach().numpy().squeeze()}')
    generation_num = 200
    print('-' * 40)
    for generation in range(generation_num):
        # s = time.time()
        rtvec = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
        # if ((rtvec.cpu().detach().numpy().squeeze() - bound[:, 0]) < 0).any() or ((rtvec.cpu().detach().numpy().squeeze() - bound[:, 1]) > 0).any():
        #     print('rtvec out of bound')
        #     break
        # pose = rtvec2pose(rtvec, norm_factor, device).cpu().detach().numpy().squeeze()
        # mTRE = cal_mTRE(CT_vol, pose_gt, pose, BATCH_SIZE, device).cpu().detach().numpy().squeeze()
        # print(f'pose: {pose} mTRE: {mTRE}')
        loss = similarity_measure(rtvec, target, CT_vol, ray_proj_mov, corner_pt, param, metric)
        # e = time.time()
        # print(f'epoch: {e - s}')
        if loss < min_loss:
            rtvec_res = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
            min_loss = loss
            min_generation = generation

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()
        optimizer6.step()
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()
        scheduler6.step()
        
        if min_generation + 100 < generation:
            print('early stop at generation:', generation)
            break
        # e = time.time()
        # print(f'epoch: {e - s}')
        
    end = time.time()
    running_time= end - start
    
    pose_res = rtvec2pose(rtvec_res, norm_factor, device).cpu().detach().numpy().squeeze()
   
    ini_mTRE = cal_mTRE(CT_vol, pose_gt, pose_init, BATCH_SIZE, device).cpu().detach().numpy().squeeze()
    res_mTRE = cal_mTRE(CT_vol, pose_gt, pose_res, BATCH_SIZE, device).cpu().detach().numpy().squeeze()

    print('min_generation: ', min_generation)
    print("total time: ", running_time, 's')
    print('pose_gt:', pose_gt)
    print('pose_ini:', pose_init)
    print('pose:', pose_res)
    print('initial mTRE: ', ini_mTRE)
    print('result mTRE: ', res_mTRE)
    return pose_init, ini_mTRE, pose_gt, pose_res, res_mTRE, np.array([0]), np.array([0]), running_time, min_generation, min_loss.cpu().detach().numpy().squeeze()

def Gradient_Descent_interface(pose_gt, rtvec_init, norm_factor, device, target, CT_vol, ray_proj_mov, corner_pt, param, metric='msp_ncc'):
    torch.cuda.empty_cache()
    pose_init = rtvec2pose(rtvec_init, norm_factor, device).cpu().detach().numpy().squeeze()
    
    rtvec_rot1 = rtvec_init[:, :1].clone().detach()
    rtvec_rot2 = rtvec_init[:, 1:2].clone().detach()
    rtvec_rot3 = rtvec_init[:, 2:3].clone().detach()
    rtvec_trans1 = rtvec_init[:, 3:4].clone().detach()
    rtvec_trans2 = rtvec_init[:, 4:5].clone().detach()
    rtvec_trans3 = rtvec_init[:, 5:].clone().detach()
    rtvec_rot1.requires_grad = True
    rtvec_rot2.requires_grad = True
    rtvec_rot3.requires_grad = True
    rtvec_trans1.requires_grad = True
    rtvec_trans2.requires_grad = True
    rtvec_trans3.requires_grad = True
    bound=[[60 * math.pi / 180, 120 * math.pi / 180],
           [-35 * math.pi / 180, 30 * math.pi / 180],
           [-30 * math.pi / 180, 30 * math.pi / 180],
           [-50 / norm_factor, 50 / norm_factor],
           [-50 / norm_factor, 50 / norm_factor],
           [650 / norm_factor, 750 / norm_factor]]
    bound = np.array(bound)

    lr_rot = 5e-2
    lr_trans = 1e-2
    lr1 = lr_rot
    lr2 = lr_rot
    lr3 = lr_rot
    lr4 = lr_trans
    lr5 = lr_trans
    lr6 = lr_trans
    optimizer1 = SGDWithBounds(params=[rtvec_rot1], bounds=bound[0], lr=lr1, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer2 = SGDWithBounds(params=[rtvec_rot2], bounds=bound[1], lr=lr2, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer3 = SGDWithBounds(params=[rtvec_rot3], bounds=bound[2], lr=lr3, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer4 = SGDWithBounds(params=[rtvec_trans1], bounds=bound[3], lr=lr4, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer5 = SGDWithBounds(params=[rtvec_trans2], bounds=bound[4], lr=lr5, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    optimizer6 = SGDWithBounds(params=[rtvec_trans3], bounds=bound[5], lr=lr6, momentum=0.6, dampening=0.45, weight_decay=1e-8)
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=25, gamma=0.9)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=25, gamma=0.9)
    scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=25, gamma=0.9)
    scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=25, gamma=0.9)
    scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=25, gamma=0.9)
    scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=25, gamma=0.9)
    start = time.time()
    rtvec_res = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
    min_generation = 0
    min_loss = similarity_measure(torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1), target, CT_vol, ray_proj_mov, corner_pt, param, metric)

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    optimizer5.zero_grad()
    optimizer6.zero_grad()
    min_loss.backward()
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    optimizer6.step()
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()
    scheduler6.step()

    # print(f'pose_gt: {rtvec2pose(rtvec_gt, norm_factor, device).cpu().detach().numpy().squeeze()}')
    # print(f'pose_init: {rtvec2pose(rtvec_init, norm_factor, device).cpu().detach().numpy().squeeze()}')
    generation_num = 200
    # print('-' * 40)
    for generation in range(generation_num):
        rtvec = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
        loss = similarity_measure(rtvec, target, CT_vol, ray_proj_mov, corner_pt, param, metric)
        if loss < min_loss:
            rtvec_res = torch.cat((rtvec_rot1, rtvec_rot2, rtvec_rot3, rtvec_trans1, rtvec_trans2, rtvec_trans3), dim=1)
            min_loss = loss
            min_generation = generation

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()
        optimizer6.step()
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        scheduler4.step()
        scheduler5.step()
        scheduler6.step()
        
        if min_generation + 100 < generation:
            print('early stop at generation:', generation)
            break
        
    end = time.time()
    running_time= end - start
    
    pose_res = rtvec2pose(rtvec_res, norm_factor, device).cpu().detach().numpy().squeeze()
    return pose_init, pose_gt, pose_res, running_time


if __name__ == "__main__":
    DF_PATH = './test_gdo_test.csv'
    eval(Gradient_Descent, 10, DF_PATH, include_ini=True, include_gen=True, model='msp_ncc')
