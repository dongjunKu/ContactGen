import sys
import os
import os.path as osp
import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import smplx
from ci3d import CI3D
from model import BaseCDDPM, BaseNet, BaseGuideNet
from params import ParamsSampleGuideNet, ParamsSampleDiffusion
from utils import save_pkl, smplx_utils


# dataset
dataset_test = CI3D(ParamsSampleGuideNet.dataset_dir, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=ParamsSampleGuideNet.batch_size, shuffle=True, num_workers=1)

# models
if ParamsSampleGuideNet.w_diffusion:
    basenet = BaseNet().to(ParamsSampleDiffusion.device) # ci3d_X5_1024
    basecddpm = BaseCDDPM(basenet, noise_steps=ParamsSampleDiffusion.noise_steps, beta_start=ParamsSampleDiffusion.beta_start, beta_end=ParamsSampleDiffusion.beta_end).to(ParamsSampleDiffusion.device)
baseguidenet = BaseGuideNet(ParamsSampleGuideNet.num_label).to(ParamsSampleGuideNet.device)

if ParamsSampleGuideNet.w_diffusion:
    # load diffusion ckpt
    ckpt_file = osp.join(ParamsSampleDiffusion.ckpt_dir, f'epoch{ParamsSampleDiffusion.load_epoch:06d}.pt')
    ckpt_files = glob.glob(osp.join(ParamsSampleDiffusion.ckpt_dir, '*.pt'))
    if len(ckpt_files) > 0:
        ckpt_file = sorted(ckpt_files)[-1] if ckpt_file is None else ckpt_file # recent ckpt
        checkpoint = torch.load(ckpt_file)
        basenet.load_state_dict(checkpoint['model_state_dict'])
        print(f'diffusion load success')
    else:
        print(f'diffusion load fail...')
        exit()

# load guidenet ckpt
ckpt_file = osp.join(ParamsSampleGuideNet.ckpt_dir, f'epoch{ParamsSampleGuideNet.load_epoch:06d}.pt')
ckpt_files = glob.glob(osp.join(ParamsSampleGuideNet.ckpt_dir, '*.pt'))
if len(ckpt_files) > 0:
    ckpt_file = sorted(ckpt_files)[-1] if ckpt_file is None else ckpt_file # recent ckpt
    checkpoint = torch.load(ckpt_file)
    try:
        baseguidenet.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        pass
    epoch = checkpoint['epoch']
    loss = checkpoint['total_loss']
    print(f'guidance load success')
else:
    print(f'guidance load fail...')
    exit()

baseguidenet.eval()
if ParamsSampleGuideNet.w_diffusion:
    basenet.eval()

interaction_types = ['Grab', 'Handshake', 'Hit', 'HoldingHands', 'Hug', 'Kick', 'Posing', 'Push']

for step, data in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):

    human_transl_gt = data['camera_wearer_transl'].to(ParamsSampleGuideNet.device)
    human_rot3_gt = data['camera_wearer_global_orient'].to(ParamsSampleGuideNet.device)
    human_pose_gt = data['camera_wearer_pose'].to(ParamsSampleGuideNet.device)
    human_lh_pose_gt = data['camera_wearer_left_hand_pose'].to(ParamsSampleGuideNet.device)
    human_rh_pose_gt = data['camera_wearer_right_hand_pose'].to(ParamsSampleGuideNet.device)
    partner_transl = data['interactee_transl'].to(ParamsSampleGuideNet.device)
    partner_rot3 = data['interactee_global_orient'].to(ParamsSampleGuideNet.device)
    partner_pose = data['interactee_pose'].to(ParamsSampleGuideNet.device)
    partner_lh_pose = data['interactee_left_hand_pose'].to(ParamsSampleGuideNet.device)
    partner_rh_pose = data['interactee_right_hand_pose'].to(ParamsSampleGuideNet.device)
    interaction_type = data['interaction_type'].to(ParamsSampleGuideNet.device)
    signature = data['signature'].to(ParamsSampleGuideNet.device)

    human_feature_gt = smplx_utils.encode(human_transl_gt, human_rot3_gt, human_pose_gt, human_lh_pose_gt, human_rh_pose_gt)
    partner_feature = smplx_utils.encode(partner_transl, partner_rot3, partner_pose, partner_lh_pose, partner_rh_pose)

    with torch.no_grad():
        if ParamsSampleGuideNet.w_diffusion:
            noise_pred, noise, human_feature_pred, t = basecddpm(human_feature_gt, partner_feature, interaction_type, return_x_pred=True, return_t=True)
        else:
            human_feature_pred = human_feature_gt

        human_params_pred = smplx_utils.decode(human_feature_pred, return_dict=True)
        human_smplx_pred = smplx_utils.smplx(**human_params_pred)
        partner_smplx = smplx_utils.smplx(transl=partner_transl, global_orient=partner_rot3, body_pose=partner_pose, left_hand_pose=partner_lh_pose, right_hand_pose=partner_rh_pose)

        signature_pred, x_segmentation_pred, cond_segmentation_pred = baseguidenet(human_smplx_pred, partner_smplx, interaction_type, t)
        signature_pred_idx = baseguidenet.sigmark2sigidx(baseguidenet.sig2sigmark(signature_pred, x_segmentation_pred, cond_segmentation_pred, thres=0.5, at_least_one=True))

    human_transl_pred, human_rot3_pred, human_pose_pred, human_lh_pose_pred, human_rh_pose_pred = smplx_utils.decode(human_feature_pred)

    interaction_type_string = interaction_types[interaction_type.item()]

    if not osp.exists(ParamsSampleGuideNet.out_dir):
        os.mkdir(ParamsSampleGuideNet.out_dir)
    save_pkl({'transl': human_transl_pred,
                'global_orient': human_rot3_pred,
                'pose': human_pose_pred,
                'left_hand_pose': human_lh_pose_pred,
                'right_hand_pose': human_rh_pose_pred,
                'signature': signature_pred_idx}, outpath=osp.join(ParamsSampleGuideNet.out_dir, f'step{step:04d}_{interaction_type_string}_human_gt'))
    save_pkl({'transl': partner_transl,
                'global_orient': partner_rot3,
                'pose': partner_pose,
                'left_hand_pose': partner_lh_pose,
                'right_hand_pose': partner_rh_pose,}, outpath=osp.join(ParamsSampleGuideNet.out_dir, f'step{step:04d}_{interaction_type_string}_partner'))
