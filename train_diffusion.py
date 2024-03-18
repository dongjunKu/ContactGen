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

from ci3d import CI3D
from model import BaseCDDPM, BaseNet
from params import ParamsTrainDiffusion
from utils import smplx_utils


# dataset
dataset_train = CI3D(ParamsTrainDiffusion.dataset_dir, mode='train')
dataloader_train = DataLoader(dataset_train, ParamsTrainDiffusion.batch_size, shuffle=True, num_workers=16, drop_last=True)

# models
basenet = BaseNet().to(ParamsTrainDiffusion.device)
basecddpm = BaseCDDPM(basenet, noise_steps=ParamsTrainDiffusion.noise_steps, beta_start=ParamsTrainDiffusion.beta_start, beta_end=ParamsTrainDiffusion.beta_end).to(ParamsTrainDiffusion.device)

# optimizer
optimizer = Adam(basenet.parameters(), lr=ParamsTrainDiffusion.learning_rate)

# loss
def baseloss(x, x_rec):
    loss_transl = F.mse_loss(x[:, :3], x_rec[:, :3])
    loss_rot6 = F.mse_loss(x[:, 3:9], x_rec[:, 3:9])
    loss_pose6 = F.mse_loss(x[:, 9:9+21*6], x_rec[:, 9:9+21*6])
    loss_h_pose6 = F.mse_loss(x[:, 9+21*6:], x_rec[:, 9+21*6:])
    return loss_transl, loss_rot6, loss_pose6, loss_h_pose6

# load ckpt
start_epoch = 0
ckpt_files = glob.glob(osp.join(ParamsTrainDiffusion.ckpt_dir, '*.pt'))
if len(ckpt_files) > 0:
    ckpt_file = sorted(ckpt_files)[-1]
    checkpoint = torch.load(ckpt_file)
    basenet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'load success | start epoch: {start_epoch}')
else:
    print(f'load fail...')
    
basenet.train()

for epoch in range(start_epoch, ParamsTrainDiffusion.max_epochs):
    for step, data in enumerate(dataloader_train):

        human_transl_gt = data['camera_wearer_transl'].to(ParamsTrainDiffusion.device)
        human_rot3_gt = data['camera_wearer_global_orient'].to(ParamsTrainDiffusion.device)
        human_pose_gt = data['camera_wearer_pose'].to(ParamsTrainDiffusion.device)
        human_lh_pose_gt = data['camera_wearer_left_hand_pose'].to(ParamsTrainDiffusion.device)
        human_rh_pose_gt = data['camera_wearer_right_hand_pose'].to(ParamsTrainDiffusion.device)
        partner_transl = data['interactee_transl'].to(ParamsTrainDiffusion.device)
        partner_rot3 = data['interactee_global_orient'].to(ParamsTrainDiffusion.device)
        partner_pose = data['interactee_pose'].to(ParamsTrainDiffusion.device)
        partner_lh_pose = data['interactee_left_hand_pose'].to(ParamsTrainDiffusion.device)
        partner_rh_pose = data['interactee_right_hand_pose'].to(ParamsTrainDiffusion.device)
        interaction_type = data['interaction_type'].to(ParamsTrainDiffusion.device)

        human_feature_gt = smplx_utils.encode(human_transl_gt, human_rot3_gt, human_pose_gt, human_lh_pose_gt, human_rh_pose_gt)
        partner_feature = smplx_utils.encode(partner_transl, partner_rot3, partner_pose, partner_lh_pose, partner_rh_pose)

        if ParamsTrainDiffusion.cfg_scale1 > 0 and np.random.random() < 0.2:
            partner_feature = None
            if ParamsTrainDiffusion.cfg_scale2 > 0 and np.random.random() < 0.5:
                interaction_type = None

        optimizer.zero_grad()
        noise_pred, noise = basecddpm(human_feature_gt, partner_feature, interaction_type)
        loss_transl, loss_rot6, loss_pose6, loss_h_pose6 = baseloss(noise_pred, noise)
        total_loss = loss_transl + loss_rot6 + loss_pose6 + loss_h_pose6
        total_loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}/{ParamsTrainDiffusion.max_epochs} | step {step}/{len(dataloader_train)} | total_loss: {total_loss.item():.4f} | loss_transl: {loss_transl.item():.4f} | loss_rot6: {loss_rot6.item():.4f} | loss_pose6: {loss_pose6.item():.4f} | loss_h_pose6: {loss_h_pose6.item():.4f}')

    if (epoch + 1) % ParamsTrainDiffusion.save_epoch == 0:
        if not osp.exists(ParamsTrainDiffusion.ckpt_dir):
            os.mkdir(ParamsTrainDiffusion.ckpt_dir)
        # save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': basenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_loss': total_loss,
            'loss_transl': loss_transl,
            'loss_rot6': loss_rot6,
            'loss_pose6': loss_pose6,
            'loss_h_pose6': loss_h_pose6
        }, osp.join(ParamsTrainDiffusion.ckpt_dir, f'epoch{epoch + 1:06d}.pt'))
