import os
import torch
import time
from net.ProST import ProST_opt
from util import input_param, init_rtvec, seed_everything,cal_mTRE,rtvec2pose,eval
from metric import  gradncc,ncc,ngi,nccl
import numpy as np
import math
from cmaes import CMA
DATA_SET = './sample/ct/256'
ct_name='test.nii.gz'
BATCH_SIZE=1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
zFlip = False
proj_size = 256
flag = 2
generation_num=50
CT_PATH = DATA_SET + ct_name
drr_generator = ProST_opt().to(device)


def similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric='ncc'):
    with torch.no_grad():
        moving = drr_generator(CT_vol, ray_proj_mov, rtvec, corner_pt, param)
    if metric=='ncc':
        return ncc(target,moving)
    elif metric=='gc':
        return gradncc(target,moving)
    elif metric=='ngi':
        return ngi(target,moving)
    elif metric=='nccl':
        return nccl(target,moving)
    else:
        raise ValueError(
            f"{metric} not recongnized, must be  ['ncc', 'gc', 'ngi'...]"
        )
        





def CMA_ES(metric='ncc',sigma=1,lr_adapt=False ):
    param, det_size, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, BATCH_SIZE,flag, proj_size )
    __, _, rtvec, rtvec_gt  = init_rtvec(BATCH_SIZE, device, norm_factor,iterative=True)
    bound=[[70*math.pi/180,110*math.pi/180],[-20*math.pi/180,20*math.pi/180],[-20*math.pi/180,20*math.pi/180],[-30/norm_factor,30/norm_factor],[-30/norm_factor,30/norm_factor],[670/norm_factor,730/norm_factor]]
    bound=np.array(bound)
    with torch.no_grad():
        target = drr_generator(CT_vol, ray_proj_mov, rtvec_gt, corner_pt, param)
    # print(bound)
    initial_rtvec=rtvec.cpu().detach().numpy().squeeze()
    # print(initial_rtvec)
    start=time.time()
    min_value=similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
    result=rtvec
    optimizer = CMA(mean=initial_rtvec, sigma=sigma,bounds=bound,lr_adapt=lr_adapt)
    for generation in range(generation_num):
        solutions = []
        # if(generation==generation_num-1):
        #     value_list=[]
        #     rtvec_list=[]
        #     rtvec_list.append(initial_rtvec)
        #     value_list.append(min_value)
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x=torch.unsqueeze(torch.tensor(x, dtype=torch.float, requires_grad=True,
                      device=device),0)
            value = similarity_measure(x,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
            solutions.append((x.cpu().detach().numpy().squeeze(), value.cpu().detach().numpy().squeeze()))
            if(min_value>=value):
                result=x
                min_value=value
            # if(generation==generation_num-1):
            #     value_list.append(value.cpu().detach().numpy().squeeze())
            #     rtvec_list.append(x.cpu().detach().numpy().squeeze())
            
            # print(f"#{generation} {value} (pose={x})")
        # if(generation==generation_num-1):
        #     i=value_list.index(min(value_list))
        #     print(value_list[i])
        #     result=rtvec_list[i]
            
        optimizer.tell(solutions)
        
        if optimizer.should_stop():
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
    print("total time: ", running_time, 's')
    print('initial_value:', rtvec_in_param)
    print('target_value:', rtvec_gt_param)
    print('initial mTRE: ', ini_mTRE)
    print('result:', rtvec_param)
    print('result mTRE: ', mTRE)
    return rtvec_in_param, ini_mTRE,rtvec_gt_param, rtvec_param, mTRE,  np.array([0]), np.array([0]), running_time



if __name__ == "__main__":
    seed = 772512  # 182501, 852097, 881411
    print('seed:', seed)
    seed_everything(seed)
    DF_PATH = './save_result/csv/test_CMA_ES.csv'
    eval(CMA_ES,100,DF_PATH,include_ini=True)
    # CMA_ES(metric='gc',sigma=0.3,lr_adapt=True)
