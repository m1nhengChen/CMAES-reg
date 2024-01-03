from util import seed_everything
seed = 772512  # 182501, 852097, 881411
print('seed:', seed)
seed_everything(seed)
from util import input_param, init_rtvec ,cal_mTRE,rtvec2pose,eval,domain_randomization
import os
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
DATA_SET = '../Data/ct/256/test'
full_scan_path='../original_data/ct/256/test'
img_files = os.listdir(DATA_SET)
ct_list = []
for img_file in img_files:
    ct_list.append(img_file)

BATCH_SIZE=1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))
zFlip = False
proj_size = 256
flag = 4
generation_num=500
drr_generator = ProST_opt().to(device)


SAVE_PATH = './save_result/SimNet_model'

RESUME_EPOCH = 550 # -1 means training from scratch
RESUME_MODEL = SAVE_PATH + '/vali_model' + str(RESUME_EPOCH) + '.pt'
model=SimNet().to(device)
checkpoint = torch.load(RESUME_MODEL)
model.load_state_dict(checkpoint['state_dict'])


def similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric='ncc'):
    with torch.no_grad():
        moving = drr_generator(CT_vol, ray_proj_mov, rtvec, corner_pt, param)
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
        





def CMA_ES(metric='msp_ncc',sigma=0.05,lr_adapt=False ):
    name=np.random.choice(ct_list, 1)[0]
    CT_PATH = DATA_SET + '/' + name
    fc_name=name.split('_256')[0]+'.nii.gz'
    Full_PATH = full_scan_path + '/' +fc_name
    param, det_size, FC_vol, ray_proj_mov, corner_pt, norm_factor = input_param(Full_PATH, BATCH_SIZE,flag, proj_size )
    param, det_size, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, BATCH_SIZE,flag, proj_size )
    
    __, _, rtvec, rtvec_gt  = init_rtvec(BATCH_SIZE, device, norm_factor,distribution="N",center = [90, 0, 0, 900, 0, 0],manual_param_range=[10,10,10,15,15,15],iterative=True)
    bound=[[70*math.pi/180,110*math.pi/180],[-21*math.pi/180,20*math.pi/180],[-20*math.pi/180,20*math.pi/180],[-30/norm_factor,30/norm_factor],[-30/norm_factor,30/norm_factor],[870/norm_factor,930/norm_factor]]
    bound=np.array(bound)
    with torch.no_grad():
        target = drr_generator(FC_vol, ray_proj_mov, rtvec_gt, corner_pt, param)
    initial_rtvec=rtvec.cpu().detach().numpy().squeeze()
    min_generation=0
    start=time.time()
    min_value=similarity_measure(rtvec,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
    result=rtvec
    optimizer = CMA(mean=initial_rtvec, sigma=sigma,bounds=bound,lr_adapt=lr_adapt)
    for generation in range(generation_num):
        solutions = []
        append=solutions.append
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x=torch.unsqueeze(torch.tensor(x, dtype=torch.float, requires_grad=False,
                      device=device),0)
            value = similarity_measure(x,target,CT_vol, ray_proj_mov,corner_pt, param,metric)
            append((x.cpu().detach().numpy().squeeze(), value.cpu().detach().numpy().squeeze()))
            if(min_value>value):
                result=x
                min_value=value
                min_generation=generation
        
        optimizer.tell(solutions)
        
        if min_generation+50<generation:
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
    # return rtvec_in_param, ini_mTRE,rtvec_gt_param, rtvec_param, mTRE,  np.array([0]), np.array([0]), running_time,min_generation,min_value.cpu().detach().numpy().squeeze()


if __name__ == "__main__":
    DF_PATH = './save_result/csv/test_CMA_ES.csv'
    # eval(CMA_ES,500,DF_PATH,include_ini=True,include_gen=True)
    CMA_ES(metric='msp_ncc',sigma=0.01,lr_adapt=True)
