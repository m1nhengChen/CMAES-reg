import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
from util import input_param, init_rtvec, seed_everything,cal_mTRE,rtvec2pose,eval,domain_randomization,pose2rtvec
seed = 772512  # 182501, 852097, 881411
print('seed:', seed)
seed_everything(seed)
import torch
import time
from net.ProST import ProST_opt
from net.SimNet import SimNet
from metric import  gradncc,ncc,ngi,nccl,PW_NCC,MPW_NCC
from metric import MultiscaleNormalizedCrossCorrelation2d as MSP_NCC
from metric import MultiscaleGradientNormalizedCrossCorrelation2d as MSP_GC
import numpy as np
import math
from cmaes import CMA
SEG_SET = '../Data/ct/256/test'
X_RAY_SET = '../Data/x_ray/256/synthetic/test_1000'
x_ray_files = os.listdir(X_RAY_SET)
x_ray_list=[]
# x_ray_list = ['data1_frontal_noBoard_256.nii.gz','data2_frontal_board_256.nii.gz','data4_frontal_noBoard_256.nii.gz','data5_frontal_noBoard_256.nii.gz','data6_frontal_noBoard_256.nii.gz']
for x_ray_file in x_ray_files:
    x_ray_list.append(x_ray_file)
# print(x_ray_list)
# gt_list=[[98., -6.5, 1.7, 609.5, 0.7, 17.3],[94.3, -4.9, -0.3, 724.2, 4.6, -14.9],[91.5, -1.8, 3., 731.9, 19.3, -30.9],[93.1, -0.6, 1.7, 730, 6, 17.7],[96.9, -1.9, 0.1, 682.6, 26.6, -13.5]]

BATCH_SIZE=1
device = torch.device('cuda:{}'.format(device_ids[0]))
zFlip = False
proj_size = 256
flag = 2
drr_generator = ProST_opt().to(device)
import pandas as pd
include_ini=True
include_gen=False
SAVE_PATH = './save_result/SimNet_model'

RESUME_EPOCH = 550 # -1 means training from scratch
RESUME_MODEL = SAVE_PATH + '/vali_model' + str(RESUME_EPOCH) + '.pt'
model=SimNet().to(device)
checkpoint = torch.load(RESUME_MODEL)
model.load_state_dict(checkpoint['state_dict'])


def similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric='ncc'):
    with torch.no_grad():
        moving = drr_generator(CT_vol, ray_proj_mov, rtvec, corner_pt, param)
        # print(ray_proj_mov.size())
        # min_mov, _ = torch.min(moving.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        # max_mov, _ = torch.max(moving.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        # moving = (moving.reshape(BATCH_SIZE, -1) - min_mov) / (max_mov - min_mov)
        # moving = moving.reshape(BATCH_SIZE, 1, ray_proj_mov.size(2), ray_proj_mov.size(3))
    if metric=='ncc':
        return ncc(target,moving)
    elif metric=='msp_ncc':
        operator= MSP_NCC(patch_sizes=[None,13],patch_weights=[0.5,0.5])
        return 1-operator(target,moving)
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
    elif metric=='neural':
        v1=model(target)
        v2=model(moving)
        v1 = v1 / torch.norm(v1, dim=1, keepdim=True).clamp_min(1.0)
        v2 = v2 / torch.norm(v2, dim=1, keepdim=True).clamp_min(1.0)
        v1=v1.flatten(start_dim=1)
        v2=v2.flatten(start_dim=1)
        pred = torch.einsum("bi,bi->b", v1, v2)
        return pred
    elif metric=='pw_ncc':
        return PW_NCC(target,moving)
    else:
        raise ValueError(
            f"{metric} not recongnized, must be  ['ncc', 'gc', 'ngi'...]"
        )
        





def CMA_ES(seg_pt,x_ray_pt, pose_gt ,metric='msp_ncc',sigma=0.05,lr_adapt=False,early_stop=False,pop_size=None,generation_num=100):

    
    param, det_size, CT_vol, X_RAY_sli,ray_proj_mov, corner_pt, norm_factor = input_param(seg_pt, BATCH_SIZE,flag, proj_size ,X_RAY_PATH=x_ray_pt)
    target = X_RAY_sli
    __, _, rtvec,_ = init_rtvec(BATCH_SIZE, device, norm_factor,manual_pose_distribution='N',manual_pose_range=[15,15,15,25,25,25],center = [90, 0, 0, 700, 0, 0],iterative=True)
    # rtvec=pose2rtvec(pose,device,norm_factor)
    rtvec_gt=pose2rtvec(pose_gt,device,norm_factor)
    # print(norm_factor)
    #__, _, rtvec, rtvec_gt  = init_rtvec(BATCH_SIZE, device, norm_factor,center = [90, 0, 0, 900, 0, 0],iterative=True)
    bound=[[60*math.pi/180,120*math.pi/180],[-35*math.pi/180,30*math.pi/180],[-30*math.pi/180,30*math.pi/180],[-50/norm_factor,50/norm_factor],[-50/norm_factor,50/norm_factor],[650/norm_factor,750/norm_factor]]
    bound=np.array(bound)
    # with torch.no_grad():
    #     target = drr_generator(FC_vol, ray_proj_mov, rtvec_gt, corner_pt, param)
    initial_rtvec=rtvec.cpu().detach().numpy().squeeze()
    # print(initial_rtvec)
    min_generation=0
    start=time.time()
    min_value=similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
    result=rtvec
    optimizer = CMA(mean=initial_rtvec, sigma=sigma,bounds=bound,lr_adapt=lr_adapt,population_size=pop_size)
    for generation in range(generation_num):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x=torch.unsqueeze(torch.tensor(x, dtype=torch.float, requires_grad=False,
                      device=device),0)
            value = similarity_measure(x,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
            solutions.append((x.cpu().detach().numpy().squeeze(), value.cpu().detach().numpy().squeeze()))
            # if(min_value>value):
            #     # print(value)
            #     result=x
            #     min_value=value
            #     min_generation=generation
            
        optimizer.tell(solutions)
        result=torch.unsqueeze(torch.tensor(optimizer._mean, dtype=torch.float, requires_grad=False,
                      device=device),0)
        # print(result)
        if optimizer.should_stop():
            print('early stop at generation:', generation)
            break
        if min_generation+5<generation and early_stop:
            print('early stop at generation:', generation)
            break
            
    end=time.time()
    
    # record and evaluate the results
    
    initial_value = rtvec2pose(rtvec, norm_factor, device)
    rtvec_in_param = initial_value.cpu().detach().numpy().squeeze()
    
    target_value = rtvec2pose(rtvec_gt, norm_factor, device)
    rtvec_gt_param = target_value.cpu().detach().numpy().squeeze()
   
    ini_mTRE = cal_mTRE(CT_vol, rtvec_gt_param, rtvec_in_param, BATCH_SIZE, device).cpu().detach().numpy().squeeze()
    
    # result=torch.unsqueeze(torch.tensor(result, dtype=torch.float, requires_grad=True,
    #                  device=device),0)
    result = rtvec2pose(result, norm_factor, device)
    rtvec_param = result.cpu().detach().numpy().squeeze()
    
    mTRE = cal_mTRE(CT_vol, rtvec_gt_param, rtvec_param, BATCH_SIZE, device).cpu().detach().numpy().squeeze()
    running_time= end-start
    # print("total time: ", running_time, 's')
    # print('initial_value:', rtvec_in_param)
    # print('target_value:', rtvec_gt_param)
    # print('initial mTRE: ', ini_mTRE)
    # print('result:', rtvec_param)
    # print('result mTRE: ', mTRE)
    return rtvec_in_param, ini_mTRE,rtvec_gt_param, rtvec_param, mTRE,  np.array([0]), np.array([0]), running_time,min_generation,min_value.cpu().detach().numpy().squeeze()

def cmaes_reg(DF_path='../Data/save_result/RTPI/self_supervised/real_test_gd_diffpose300.csv',iter_num=5,pop_size=30,generation_num=100,sigma=1,lr_adapt=False,early_stop=False):
    # col_names = ['tar',
    #          'pred']
    # data = np.genfromtxt(DF_path,  delimiter=',',dtype=np.float32,encoding='utf-8-sig')
    # data=np.squeeze(data)
    print("test for CMA-ES on %d samples, parameter setting: \n generation: %d, population size: %s, sigma:%s, learning rate adaption: %s" %(iter_num,generation_num,pop_size,sigma,lr_adapt))
    if include_ini:
        ini_ls=[]
        ini_mTRE_ls=[]
        ini_error_ls=[]
    if include_gen:
        min_gen_ls=[]
        min_val_ls=[]
        min_val_ls.append(0.)
    mTRE_ls = []
    pred_kp_dist_ls = []
    pred_ncc_ls = []
    time_ls = []
    data_ls = []
    pred_error_ls = []
    for i in range(iter_num):
        # if data[i][10]>30:
        #     data[i][10]=30
        x_ray_name = x_ray_list[i % len(x_ray_list)]
        x_ray_split = x_ray_name.split('.nii.gz')[0].split('_')
        ct_name = x_ray_split[0]
        # pose_gt = gt_list[i% len(x_ray_list)]
        
        seg_pt = f'{SEG_SET}/{ct_name}_256.nii.gz'
        x_ray_pt = f'{X_RAY_SET}/{x_ray_name}'
        pose_gt = [float(i) for i in x_ray_split[1:7]]
        tar = np.array(pose_gt, ndmin=2, dtype=np.float32)
        tar = torch.tensor(tar, dtype=torch.float, requires_grad=False, device=device)
        # print(tar)
        # tar=torch.unsqueeze(torch.tensor(tar, dtype=torch.float, requires_grad=False,
        #               device=device),0)
        # pred=torch.unsqueeze(torch.tensor(pred, dtype=torch.float, requires_grad=False,
        #               device=device),0)
        ini,ini_mTRE,tar, pred, mTRE, pred_kp_dist, pred_ncc, t, min_g, min_v =CMA_ES(seg_pt,x_ray_pt,tar,metric='msp_ncc',sigma=sigma,lr_adapt=lr_adapt,pop_size=pop_size,generation_num=generation_num,early_stop=early_stop)
        # print(ini_mTRE)
        # print(mTRE)
        data_ls.append([ct_name, *[str(i) for i in np.append(tar, pred, axis = 0).squeeze()]])
        if include_ini:
            ini_ls.append(ini)
            ini_mTRE_ls.append(ini_mTRE)
            ini_error_ls.append(abs(ini - tar))
        if include_gen:
            min_gen_ls.append(min_g)
            if mTRE < 10:
                min_val_ls.append(min_v)
        mTRE_ls.append(mTRE)
        pred_kp_dist_ls.append(pred_kp_dist)
        pred_ncc_ls.append(pred_ncc)
        time_ls.append(t)
        pred_error_ls.append(abs(pred - tar))

    print('--------------------------------------------------------------------------------')
    data_df = pd.DataFrame(data_ls)
    data_df.to_csv(DF_path, header=False, index=False)
    if include_ini:
        ini_mTRE_array=np.array(ini_mTRE_ls)
        ini_mTRE_mean=np.mean(ini_mTRE_array)
        ini_mTRE_stddev = np.std(ini_mTRE_array)
        ini_success_rate=np.sum(ini_mTRE_array < 10.0) / iter_num
        ini_mTRE_array.sort()
        ini_mTRE_95 = np.percentile(ini_mTRE_array, 95)
        ini_mTRE_75 = np.percentile(ini_mTRE_array, 75)
        ini_mTRE_50 = np.percentile(ini_mTRE_array, 50)
        ini_mTRE_top_95_mean = np.mean(ini_mTRE_array[:int(0.95 * iter_num)])
        ini_mTRE_top_95_stddev = np.std(ini_mTRE_array[:int(0.95 * iter_num)])
        ini_mTRE_top_75_mean = np.mean(ini_mTRE_array[:int(0.75 * iter_num)])
        ini_mTRE_top_75_stddev = np.std(ini_mTRE_array[:int(0.75 * iter_num)])
        ini_mTRE_top_50_mean = np.mean(ini_mTRE_array[:int(0.50 * iter_num)])
        ini_mTRE_top_50_stddev = np.std(ini_mTRE_array[:int(0.50 * iter_num)])
        ini_error = np.array(ini_error_ls)
        ini_error_rot = np.sum(ini_error[:, :3], 1)
        ini_error_rot1 = ini_error[:, 0]
        ini_error_rot2 = ini_error[:, 1]
        ini_error_rot3 = ini_error[:, 2]
        ini_error_trans = np.sum(ini_error[:, 3:], 1)
        ini_error_trans1 = ini_error[:, 3]
        ini_error_trans2 = ini_error[:, 4]
        ini_error_trans3 = ini_error[:, 5]
        ini_mean_rot = np.mean(ini_error_rot)
        ini_mean_rot1 = np.mean(ini_error_rot1)
        ini_mean_rot2 = np.mean(ini_error_rot2)
        ini_mean_rot3 = np.mean(ini_error_rot3)
        ini_mean_trans = np.mean(ini_error_trans)
        ini_mean_trans1 = np.mean(ini_error_trans1)
        ini_mean_trans2 = np.mean(ini_error_trans2)
        ini_mean_trans3 = np.mean(ini_error_trans3)
        ini_stddev_rot = np.std(ini_error_rot)
        ini_stddev_rot1 = np.std(ini_error_rot1)
        ini_stddev_rot2 = np.std(ini_error_rot2)
        ini_stddev_rot3 = np.std(ini_error_rot3)
        ini_stddev_trans = np.std(ini_error_trans)
        ini_stddev_trans1 = np.std(ini_error_trans1)
        ini_stddev_trans2 = np.std(ini_error_trans2)
        ini_stddev_trans3 = np.std(ini_error_trans3)
        ini_median_rot = np.median(ini_error_rot)
        ini_median_rot1 = np.median(ini_error_rot1)
        ini_median_rot2 = np.median(ini_error_rot2)
        ini_median_rot3 = np.median(ini_error_rot3)
        ini_median_trans = np.median(ini_error_trans)
        ini_median_trans1 = np.median(ini_error_trans1)
        ini_median_trans2 = np.median(ini_error_trans2)
        ini_median_trans3 = np.median(ini_error_trans3)
    if include_gen:
        min_gen_array=np.array(min_gen_ls)
        min_gen_mean=np.mean(min_gen_array)
        min_gen_median=np.median(min_gen_array)
        min_gen_stddev=np.std(min_gen_array)
        min_gen_max=np.max(min_gen_array)
        min_gen_min=np.min(min_gen_array)
        min_val_array=np.array(min_val_ls)
        min_val_mean=np.mean(min_val_array)
        min_val_median=np.median(min_val_array)
        min_val_stddev=np.std(min_val_array)
        min_val_max=np.max(min_val_array)
        min_val_min=np.min(min_val_array)
        print('The generation that produces the optimal value : {0:.4f}±{1:.4f}'.format(min_gen_mean,min_gen_stddev))
        print('median: {0:.4f}'.format(min_gen_median))
        print('min/max: {0:.4f}:{1:.4f}'.format(min_gen_min, min_gen_max))
        print('The optimal value : {0:.4f}±{1:.4f}'.format(min_val_mean,min_val_stddev))
        print('median: {0:.4f}'.format(min_val_median))
        print('min/max: {0:.4f}:{1:.4f}'.format(min_val_min, min_val_max))

    mTRE_array = np.array(mTRE_ls)
    mTRE_mean = np.mean(mTRE_array)
    mTRE_stddev = np.std(mTRE_array)
    
    mTRE_array.sort()
    success_rate=np.sum(mTRE_array < 10.0)/iter_num
    mTRE_95 = np.percentile(mTRE_array, 95)
    mTRE_75 = np.percentile(mTRE_array, 75)
    mTRE_50 = np.percentile(mTRE_array, 50)
    mTRE_top_95_mean = np.mean(mTRE_array[:int(0.95 * iter_num)])
    mTRE_top_95_stddev = np.std(mTRE_array[:int(0.95 * iter_num)])
    mTRE_top_75_mean = np.mean(mTRE_array[:int(0.75 * iter_num)])
    mTRE_top_75_stddev = np.std(mTRE_array[:int(0.75 * iter_num)])
    mTRE_top_50_mean = np.mean(mTRE_array[:int(0.50 * iter_num)])
    mTRE_top_50_stddev = np.std(mTRE_array[:int(0.50 * iter_num)])
    
    pred_kp_dist_array = np.array(pred_kp_dist_ls)
    pred_kp_dist_mean = np.mean(pred_kp_dist_array)
    pred_ncc_array = np.array(pred_ncc_ls)
    pred_ncc_mean = np.mean(pred_ncc_array)
    pred_error = np.array(pred_error_ls)
    pred_error_rot = np.sum(pred_error[:, :3], 1)
    pred_error_rot1 = pred_error[:, 0]
    pred_error_rot2 = pred_error[:, 1]
    pred_error_rot3 = pred_error[:, 2]
    pred_error_trans = np.sum(pred_error[:, 3:], 1)
    pred_error_trans1 = pred_error[:, 3]
    pred_error_trans2 = pred_error[:, 4]
    pred_error_trans3 = pred_error[:, 5]
    pred_mean_rot = np.mean(pred_error_rot)
    pred_mean_rot1 = np.mean(pred_error_rot1)
    pred_mean_rot2 = np.mean(pred_error_rot2)
    pred_mean_rot3 = np.mean(pred_error_rot3)
    pred_mean_trans = np.mean(pred_error_trans)
    pred_mean_trans1 = np.mean(pred_error_trans1)
    pred_mean_trans2 = np.mean(pred_error_trans2)
    pred_mean_trans3 = np.mean(pred_error_trans3)
    pred_stddev_rot = np.std(pred_error_rot)
    pred_stddev_rot1 = np.std(pred_error_rot1)
    pred_stddev_rot2 = np.std(pred_error_rot2)
    pred_stddev_rot3 = np.std(pred_error_rot3)
    pred_stddev_trans = np.std(pred_error_trans)
    pred_stddev_trans1 = np.std(pred_error_trans1)
    pred_stddev_trans2 = np.std(pred_error_trans2)
    pred_stddev_trans3 = np.std(pred_error_trans3)
    pred_median_rot = np.median(pred_error_rot)
    pred_median_rot1 = np.median(pred_error_rot1)
    pred_median_rot2 = np.median(pred_error_rot2)
    pred_median_rot3 = np.median(pred_error_rot3)
    pred_median_trans = np.median(pred_error_trans)
    pred_median_trans1 = np.median(pred_error_trans1)
    pred_median_trans2 = np.median(pred_error_trans2)
    pred_median_trans3 = np.median(pred_error_trans3)

    time_array = np.array(time_ls)
    avg_time = np.mean(time_array)
    if include_ini:
        print('initial mTRE: {0:.4f}±{1:.4f}'.format(ini_mTRE_mean,ini_mTRE_stddev))
        print('ini_mTRE-95: {0:.4f}'.format(ini_mTRE_95))
        print('ini_mTRE-75: {0:.4f}'.format(ini_mTRE_75))
        print('ini_mTRE-50: {0:.4f}'.format(ini_mTRE_50))
        print('ini_mTRE_top-95: {0:.4f}±{1:.4f}'.format(ini_mTRE_top_95_mean, ini_mTRE_top_95_stddev))
        print('ini_mTRE_top-75: {0:.4f}±{1:.4f}'.format(ini_mTRE_top_75_mean, ini_mTRE_top_75_stddev))
        print('ini_mTRE_top-50: {0:.4f}±{1:.4f}'.format(ini_mTRE_top_50_mean, ini_mTRE_top_50_stddev))
        print('ini_success rate: {0:.4f}'.format(ini_success_rate))
        print('ini_mean_rot: {0:.4f}±{1:.4f},\nini_mean_trans: {2:.4f}±{3:.4f}'.format(ini_mean_rot,ini_stddev_rot,
                                                                                     ini_mean_trans,
                                                                                     ini_stddev_trans))
        print('ini_mean_rot1: {0:.4f}±{1:.4f},\n'
          'ini_mean_rot2: {2:.4f}±{3:.4f},\n'
          'ini_mean_rot3: {4:.4f}±{5:.4f},\n'
          'ini_mean_trans1: {6:.4f}±{7:.4f},\n'
          'ini_mean_trans2: {8:.4f}±{9:.4f},\n'
          'ini_mean_trans3: {10:.4f}±{11:.4f}'.format(ini_mean_rot1, ini_stddev_rot1, ini_mean_rot2,
                                                       ini_stddev_rot2, ini_mean_rot3, ini_stddev_rot3,
                                                       ini_mean_trans1, ini_stddev_trans1, ini_mean_trans2,
                                                       ini_stddev_trans2, ini_mean_trans3, ini_stddev_trans3))
        print('ini_median_rot: {0:.4f},\nini_median_trans: {1:.4f}'.format(ini_median_rot, ini_median_trans))
        print('ini_median_rot1: {0:.4f},\n'
            'ini_median_rot2: {1:.4f},\n'
            'ini_median_rot3: {2:.4f},\n'
            'ini_median_trans1: {3:.4f},\n'
            'ini_median_trans2: {4:.4f},\n'
            'ini_median_trans3: {5:.4f}'.format(ini_median_rot1, ini_median_rot2, ini_median_rot3, ini_median_trans1,
                                                ini_median_trans2, ini_median_trans3))
    print('mTRE: {0:.4f}±{1:.4f}'.format(mTRE_mean, mTRE_stddev))
    print('mTRE-95: {0:.4f}'.format(mTRE_95))
    print('mTRE-75: {0:.4f}'.format(mTRE_75))
    print('mTRE-50: {0:.4f}'.format(mTRE_50))
    print('mTRE_top-95: {0:.4f}±{1:.4f}'.format(mTRE_top_95_mean, mTRE_top_95_stddev))
    print('mTRE_top-75: {0:.4f}±{1:.4f}'.format(mTRE_top_75_mean, mTRE_top_75_stddev))
    print('mTRE_top-50: {0:.4f}±{1:.4f}'.format(mTRE_top_50_mean, mTRE_top_50_stddev))
    print('success rate: {0:.4f}'.format(success_rate))
    print('pred_key_point_distance_mean: {0:.4f}'.format(pred_kp_dist_mean))
    print('pred_ncc_mean: {0:.4f}'.format(pred_ncc_mean))
    print('pred_mean_rot: {0:.4f}±{1:.4f},\npred_mean_trans: {2:.4f}±{3:.4f}'.format(pred_mean_rot, pred_stddev_rot,
                                                                                     pred_mean_trans,
                                                                                     pred_stddev_trans))
    print('pred_mean_rot1: {0:.4f}±{1:.4f},\n'
          'pred_mean_rot2: {2:.4f}±{3:.4f},\n'
          'pred_mean_rot3: {4:.4f}±{5:.4f},\n'
          'pred_mean_trans1: {6:.4f}±{7:.4f},\n'
          'pred_mean_trans2: {8:.4f}±{9:.4f},\n'
          'pred_mean_trans3: {10:.4f}±{11:.4f}'.format(pred_mean_rot1, pred_stddev_rot1, pred_mean_rot2,
                                                       pred_stddev_rot2, pred_mean_rot3, pred_stddev_rot3,
                                                       pred_mean_trans1, pred_stddev_trans1, pred_mean_trans2,
                                                       pred_stddev_trans2, pred_mean_trans3, pred_stddev_trans3))
    print('pred_median_rot: {0:.4f},\npred_median_trans: {1:.4f}'.format(pred_median_rot, pred_median_trans))
    print('pred_median_rot1: {0:.4f},\n'
          'pred_median_rot2: {1:.4f},\n'
          'pred_median_rot3: {2:.4f},\n'
          'pred_median_trans1: {3:.4f},\n'
          'pred_median_trans2: {4:.4f},\n'
          'pred_median_trans3: {5:.4f}'.format(pred_median_rot1, pred_median_rot2, pred_median_rot3, pred_median_trans1,
                                               pred_median_trans2, pred_median_trans3))
    print('avg_time: {0:.4f}'.format(avg_time))
        # print(tar)
        
if __name__ == "__main__":
    cmaes_reg(DF_path='./save_result/csv/test_CMA_ES.csv',iter_num=1000,pop_size=None,sigma=20.0,generation_num=50,lr_adapt=True)
