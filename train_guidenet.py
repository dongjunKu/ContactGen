import sys
import os
import os.path as osp
import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from torch.optim import Adam

from ci3d import CI3D
from model import BaseCDDPM, BaseNet, BaseGuideNet
from params import ParamsTrainGuideNet, ParamsSampleDiffusion
from utils import smplx_utils


# dataset
dataset_train = CI3D(ParamsTrainGuideNet.dataset_dir, mode='train')
weights = dataset_train.weight4balancedsampling()
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(dataset_train))
dataloader_train = DataLoader(dataset_train, ParamsTrainGuideNet.batch_size, sampler=sampler, num_workers=16, drop_last=True)

# models
if ParamsTrainGuideNet.w_diffusion:
    basenet = BaseNet().to(ParamsSampleDiffusion.device) 
    basecddpm = BaseCDDPM(basenet, noise_steps=ParamsSampleDiffusion.noise_steps, beta_start=ParamsSampleDiffusion.beta_start, beta_end=ParamsSampleDiffusion.beta_end).to(ParamsSampleDiffusion.device)
baseguidenet = BaseGuideNet(num_classes=ParamsTrainGuideNet.num_label).to(ParamsTrainGuideNet.device)

# optimizer
optimizer = Adam(baseguidenet.parameters(), lr=ParamsTrainGuideNet.learning_rate)

# loss
def baseloss(pred, gt, sparse=False, s=10):
    
    """loss = F.binary_cross_entropy(pred, gt, reduction='none')
    loss = 16 * gt * loss + (1 - gt) * loss
    loss = torch.mean(loss)"""
    
    if sparse is False:
        loss = F.binary_cross_entropy(pred, gt)
    else:
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        if gt.ndim == 3: # (B, 75, 75)
            coeff = (s / gt.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True))
        if gt.ndim == 2: # (B, 75)
            coeff = (s / gt.sum(dim=1, keepdim=True))
        loss = coeff * gt * loss + (1 - gt) * loss
        loss = torch.mean(loss)

    return loss

if ParamsTrainGuideNet.w_diffusion:
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
start_epoch = 0
ckpt_files = glob.glob(osp.join(ParamsTrainGuideNet.ckpt_dir, '*.pt'))
if len(ckpt_files) > 0:
    ckpt_file = sorted(ckpt_files)[-1]
    checkpoint = torch.load(ckpt_file)
    baseguidenet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    print(f'guidance load success | start epoch: {start_epoch}')
else:
    print(f'guidance load fail...')

baseguidenet.train()
if ParamsTrainGuideNet.w_diffusion:
    basenet.eval()

for epoch in range(start_epoch, ParamsTrainGuideNet.max_epochs):

    epoch_loss = 0

    for step, data in enumerate(dataloader_train):

        human_transl_gt = data['camera_wearer_transl'].to(ParamsTrainGuideNet.device)
        human_rot3_gt = data['camera_wearer_global_orient'].to(ParamsTrainGuideNet.device)
        human_pose_gt = data['camera_wearer_pose'].to(ParamsTrainGuideNet.device)
        human_lh_pose_gt = data['camera_wearer_left_hand_pose'].to(ParamsTrainGuideNet.device)
        human_rh_pose_gt = data['camera_wearer_right_hand_pose'].to(ParamsTrainGuideNet.device)
        partner_transl = data['interactee_transl'].to(ParamsTrainGuideNet.device)
        partner_rot3 = data['interactee_global_orient'].to(ParamsTrainGuideNet.device)
        partner_pose = data['interactee_pose'].to(ParamsTrainGuideNet.device)
        partner_lh_pose = data['interactee_left_hand_pose'].to(ParamsTrainGuideNet.device)
        partner_rh_pose = data['interactee_right_hand_pose'].to(ParamsTrainGuideNet.device)
        interaction_type = data['interaction_type'].to(ParamsTrainGuideNet.device)
        signature = data['signature'].to(ParamsTrainGuideNet.device)
        x_segmentation = torch.clamp_max(torch.sum(signature, dim=2), max=1)
        cond_segmentation = torch.clamp_max(torch.sum(signature, dim=1), max=1)

        human_feature_gt = smplx_utils.encode(human_transl_gt, human_rot3_gt, human_pose_gt, human_lh_pose_gt, human_rh_pose_gt)
        partner_feature = smplx_utils.encode(partner_transl, partner_rot3, partner_pose, partner_lh_pose, partner_rh_pose)

        with torch.no_grad():
            if ParamsTrainGuideNet.w_diffusion:
                noise_pred, noise, human_feature_pred, t = basecddpm(human_feature_gt, partner_feature, interaction_type, return_x_pred=True, return_t=True)
            else:
                human_feature_pred = human_feature_gt
        
        human_params_pred = smplx_utils.decode(human_feature_pred, return_dict=True)
        human_smplx_pred = smplx_utils.smplx(**human_params_pred)
        partner_smplx = smplx_utils.smplx(transl=partner_transl, global_orient=partner_rot3, body_pose=partner_pose, left_hand_pose=partner_lh_pose, right_hand_pose=partner_rh_pose)
        
        optimizer.zero_grad()
        signature_pred, x_segmentation_pred, cond_segmentation_pred = baseguidenet(human_smplx_pred, partner_smplx, interaction_type, t)
        loss_sig = baseloss(signature_pred, signature, sparse=True, s=10)
        loss_x_seg = baseloss(x_segmentation_pred, x_segmentation, sparse=False)
        loss_cond_seg = baseloss(cond_segmentation_pred, cond_segmentation, sparse=False)
        total_loss = loss_sig + loss_x_seg + loss_cond_seg
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss / len(dataloader_train)

        print(f'Epoch {epoch}/{ParamsTrainGuideNet.max_epochs} | step {step}/{len(dataloader_train)} | loss_sig: {loss_sig.item():.4f} | loss_x_seg: {loss_x_seg.item():.4f} | loss_cond_seg: {loss_cond_seg.item():.4f}')

    print(f'--------Epoch {epoch}/{ParamsTrainGuideNet.max_epochs} | epoch_loss: {epoch_loss.item():.4f}--------')

    if (epoch + 1) % ParamsTrainGuideNet.save_epoch == 0:
        if not osp.exists(ParamsTrainGuideNet.ckpt_dir):
            os.mkdir(ParamsTrainGuideNet.ckpt_dir)
        # save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': baseguidenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': total_loss,
        }, osp.join(ParamsTrainGuideNet.ckpt_dir, f'epoch{epoch + 1:06d}.pt'))
