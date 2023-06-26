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
from egobody import Egobody
from ci3d import CI3D
from model import BaseCDDPM, BaseNetX3, BaseNetX4, BaseNetX5, GeometryTransformer

# sysargv
name = 'ci3d_X5_1024_step1000_5e-3' # ci3d_X4_512
ckpt_dir = f'checkpoint_{name}'
dataset_dir = '/workspace/dataset/ci3d' # '/workspace/dataset/egobody'
smplx_dir = 'body_visualizer/support_data/downloads/models'
epochs = 100000
start_epoch = 0
batch_size = 32
learning_rate = 1e-4
cfg_scale1 = 0.5
cfg_scale2 = 3
vposer_ckpt_dir = 'V02_05'
device = 'cuda'

# dataset
dataset_train = CI3D(dataset_dir, mode='train') # Egobody(dataset_dir, mode='train')
dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, num_workers=32)
# dataset_test = CI3D(dataset_dir, mode='test') # Egobody(dataset_dir, mode='test')
# dataloader_test = DataLoader(dataset_test, 1, shuffle=True, num_workers=32)

# models
basenet = BaseNetX5().to(device) # ci3d_X5_1024
# basecddpm = BaseCDDPM(basenet, noise_steps=1000, beta_start=1e-5, beta_end=1e-2).to(device)
basecddpm = BaseCDDPM(basenet, noise_steps=1000, beta_start=5e-6, beta_end=5e-3).to(device)
# basecddpm = BaseCDDPM(basenet).to(device)
body_model = smplx.create(model_path=smplx_dir,
                            model_type='smplx',
                            gender='neutral',
                            use_pca=False,
                            batch_size=batch_size).to(device)

# optimizer
optimizer = Adam(basenet.parameters(), lr=learning_rate)

# loss
def baseloss(x, x_rec):
    loss_transl = F.mse_loss(x[:, :3], x_rec[:, :3])
    loss_rot6 = F.mse_loss(x[:, 3:9], x_rec[:, 3:9])
    loss_pose6 = F.mse_loss(x[:, 9:9+21*6], x_rec[:, 9:9+21*6])
    loss_h_pose6 = F.mse_loss(x[:, 9+21*6:], x_rec[:, 9+21*6:])
    return loss_transl, loss_rot6, loss_pose6, loss_h_pose6

# load ckpt
ckpt_files = glob.glob(osp.join(ckpt_dir, '*.pt'))
if len(ckpt_files) > 0:
    ckpt_file = sorted(ckpt_files)[-1]
    checkpoint = torch.load(ckpt_file)
    basenet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    print(f'load success | start epoch: {start_epoch}')
else:
    print(f'load fail...')
    
basenet.train()

for epoch in range(start_epoch, epochs):
    for step, data in enumerate(dataloader_train):

        human_transl_gt = data['camera_wearer_transl'].to(device)
        human_rot3_gt = data['camera_wearer_global_orient'].to(device)
        human_pose_gt = data['camera_wearer_pose'].to(device)
        human_lh_pose_gt = data['camera_wearer_left_hand_pose'].to(device)
        human_rh_pose_gt = data['camera_wearer_right_hand_pose'].to(device)
        partner_transl = data['interactee_transl'].to(device)
        partner_rot3 = data['interactee_global_orient'].to(device)
        partner_pose = data['interactee_pose'].to(device)
        partner_lh_pose = data['interactee_left_hand_pose'].to(device)
        partner_rh_pose = data['interactee_right_hand_pose'].to(device)
        interaction_type = data['interaction_type'].to(device)

        """
        # train one label
        label = 5 # kick
        mask = (interaction_type.squeeze(dim=1) == label)
        if torch.sum(mask).item() == 0:
            continue
        human_transl_gt = human_transl_gt[mask]
        human_rot3_gt = human_rot3_gt[mask]
        human_pose_gt = human_pose_gt[mask]
        partner_transl = partner_transl[mask]
        partner_rot3 = partner_rot3[mask]
        partner_pose = partner_pose[mask]
        interaction_type = interaction_type[mask]
        """

        # if human_transl_gt.shape[0] != batch_size:
        #     continue

        human_rot6_gt = GeometryTransformer.convert_to_6D_rot(human_rot3_gt)
        human_pose6_gt = GeometryTransformer.convert_to_6D_rot(human_pose_gt).reshape(-1, 21 * 6)
        human_lh_pose6_gt = GeometryTransformer.convert_to_6D_rot(human_lh_pose_gt).reshape(-1, 15 * 6)
        human_rh_pose6_gt = GeometryTransformer.convert_to_6D_rot(human_rh_pose_gt).reshape(-1, 15 * 6)
        human_feature_gt = torch.cat([human_transl_gt, human_rot6_gt, human_pose6_gt, human_lh_pose6_gt, human_rh_pose6_gt], axis=-1)

        partner_rot6 = GeometryTransformer.convert_to_6D_rot(partner_rot3)
        partner_pose6 = GeometryTransformer.convert_to_6D_rot(partner_pose).reshape(-1, 21 * 6)
        partner_lh_pose6_gt = GeometryTransformer.convert_to_6D_rot(partner_lh_pose).reshape(-1, 15 * 6)
        partner_rh_pose6_gt = GeometryTransformer.convert_to_6D_rot(partner_rh_pose).reshape(-1, 15 * 6)
        partner_feature = torch.cat([partner_transl, partner_rot6, partner_pose6, partner_lh_pose6_gt, partner_rh_pose6_gt], axis=-1)

        if cfg_scale1 > 0 and np.random.random() < 0.2:
            partner_feature = None
            if cfg_scale2 > 0 and np.random.random() < 0.5:
                interaction_type = None

        optimizer.zero_grad()
        noise_pred, noise = basecddpm(human_feature_gt, partner_feature, interaction_type)
        # noise_pred, noise, human_feature_pred = basecddpm(human_feature_gt, partner_feature, interaction_type, return_x_pred=True)
        loss_transl, loss_rot6, loss_pose6, loss_h_pose6 = baseloss(noise_pred, noise)
        # loss_smplx = smplx_loss(human_feature_gt, human_feature_pred)
        total_loss = loss_transl + loss_rot6 + loss_pose6 + loss_h_pose6
        # total_loss = loss_smplx
        total_loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}/{epochs} | step {step}/{len(dataloader_train)} | total_loss: {total_loss.item():.4f} | loss_transl: {loss_transl.item():.4f} | loss_rot6: {loss_rot6.item():.4f} | loss_pose6: {loss_pose6.item():.4f} | loss_h_pose6: {loss_h_pose6.item():.4f}') # | loss_smplx: {loss_smplx.item():.4f}')
        # print(f'Epoch {epoch}/{epochs} | step {step}/{len(dataloader_train)} | total_loss: {total_loss.item():.4f} | loss_smplx: {loss_smplx.item():.4f}')

    if (epoch + 1) % 1 == 0:
        if not osp.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
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
        }, osp.join(ckpt_dir, f'epoch{epoch + 1:06d}.pt'))
