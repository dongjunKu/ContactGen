import sys
import os
import os.path as osp
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

from egobody import Egobody
from ci3d import CI3D
from model import BaseCDDPM, BaseNetX3, BaseNetX4, BaseNetX5, GeometryTransformer
from optimizer import BaseOptimizer
from utils import save_pkl, save_batch_pkl

# sysargv
name = 'ci3d_X5_1024_step1000_5e-3'
comment = '_momentum0.9_noise0.5_nonrm_balanced_predflow'
epoch = 100
ckpt_dir = f'checkpoint_{name}'
dataset_dir = '/workspace/dataset/ci3d' # '/workspace/dataset/egobody'
smplx_dir = '/workspace/dataset/models_smplx_v1_1/models'
num_samples = 5
cfg_scale1 = 0.5
cfg_scale2 = 3
vposer_ckpt_dir = 'V02_05'
device = 'cuda'
num_label = 8
max_sample_per_label = 10 # s10

# history
history = [max_sample_per_label] * num_label

# dataset
dataset_test = CI3D(dataset_dir, mode='test') # Egobody(dataset_dir, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)

# models
basenet = BaseNetX5().to(device) # ci3d_X5_1024
baseopt = BaseOptimizer(model_path=smplx_dir, batch_size=1 * num_samples, device=device)
basecddpm = BaseCDDPM(model=basenet, optim=baseopt, noise_steps=1000, beta_start=1e-5, beta_end=1e-2).to(device)
# basecddpm = BaseCDDPM(model=basenet, optim=baseopt).to(device)

# load ckpt
ckpt_file = osp.join(ckpt_dir, f'epoch{epoch:06d}.pt')
# ckpt_file = None

ckpt_files = glob.glob(osp.join(ckpt_dir, '*.pt'))
if len(ckpt_files) > 0:
    ckpt_file = sorted(ckpt_files)[-1] if ckpt_file is None else ckpt_file # recent ckpt
    checkpoint = torch.load(ckpt_file)
    basenet.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['total_loss']
    print(f'load success')
else:
    print(f'load fail...')

out_dir = f'output_{name}_epoch{epoch}_loss{loss:.5f}' + comment

for step, data in enumerate(dataloader_test):

    if step % 20 != 0:
        continue

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
    contact_gt = data['contact']
    print('contact_gt.shape:', contact_gt.shape)
    print('(contact_gt == 10475)[:,0].all():', (contact_gt == 10475)[:,0].all())
    print('(contact_gt == 10475)[:,1].all():', (contact_gt == 10475)[:,1].all())
    print(contact_gt)

    label = interaction_type.cpu().item()
    if history[label] == 0:
        continue
    else:
        history[label] = history[label] - 1

    # interaction_type[:] = 0

    partner_rot6 = GeometryTransformer.convert_to_6D_rot(partner_rot3)
    partner_pose6 = GeometryTransformer.convert_to_6D_rot(partner_pose).reshape(-1, 21 * 6)
    partner_lh_pose6_gt = GeometryTransformer.convert_to_6D_rot(partner_lh_pose).reshape(-1, 15 * 6)
    partner_rh_pose6_gt = GeometryTransformer.convert_to_6D_rot(partner_rh_pose).reshape(-1, 15 * 6)
    partner_feature = torch.cat([partner_transl, partner_rot6, partner_pose6, partner_lh_pose6_gt, partner_rh_pose6_gt], axis=-1)

    opt_dict = {'partner_param': {
                    'transl': partner_transl,
                    'global_orient': partner_rot3,
                    'body_pose': partner_pose,
                    'left_hand_pose': partner_lh_pose,
                    'right_hand_pose': partner_rh_pose
                },
                'contact': contact_gt}

    human_feature_pred, human_feature_steps = basecddpm.sample(partner_feature, interaction_type, num_samples, cfg_scale1, cfg_scale2, opt_dict)

    human_transl_pred, human_rot6_pred, human_pose6_pred, human_lh_pose6_pred, human_rh_pose6_pred = \
        human_feature_pred[:, :3], human_feature_pred[:, 3:9], human_feature_pred[:, 9:9+21*6], human_feature_pred[:, 9+21*6:9+21*6+15*6], human_feature_pred[:, 9+21*6+15*6:]
    human_transl_steps, human_rot6_steps, human_pose6_steps, human_lh_pose6_steps, human_rh_pose6_steps = \
        human_feature_steps[:, :, :3], human_feature_steps[:, :, 3:9], human_feature_steps[:, :, 9:9+21*6], human_feature_steps[:, :, 9+21*6:9+21*6+15*6], human_feature_steps[:, :, 9+21*6+15*6:]

    human_rot3_pred = GeometryTransformer.convert_to_3D_rot(human_rot6_pred.view(-1, 6)).view(num_samples, 3)
    human_pose_pred = GeometryTransformer.convert_to_3D_rot(human_pose6_pred.reshape(-1, 6)).view(num_samples, 63)
    human_lh_pose_pred = GeometryTransformer.convert_to_3D_rot(human_lh_pose6_pred.reshape(-1, 6)).view(num_samples, 15*3)
    human_rh_pose_pred = GeometryTransformer.convert_to_3D_rot(human_rh_pose6_pred.reshape(-1, 6)).view(num_samples, 15*3)

    human_rot3_steps = GeometryTransformer.convert_to_3D_rot(human_rot6_steps.view(-1, 6)).view(num_samples, -1, 3)
    human_pose_steps = GeometryTransformer.convert_to_3D_rot(human_pose6_steps.reshape(-1, 6)).view(num_samples, -1, 63)
    human_lh_pose_steps = GeometryTransformer.convert_to_3D_rot(human_lh_pose6_steps.reshape(-1, 6)).view(num_samples, -1, 15*3)
    human_rh_pose_steps = GeometryTransformer.convert_to_3D_rot(human_rh_pose6_steps.reshape(-1, 6)).view(num_samples, -1, 15*3)

    interaction_type_string = dataset_test.interaction_types[interaction_type.item()]

    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    save_pkl({'transl': human_transl_gt,
                'global_orient': human_rot3_gt,
                'pose': human_pose_gt,
                'left_hand_pose': human_lh_pose_gt,
                'right_hand_pose': human_rh_pose_gt,
                'contact': contact_gt}, outpath=osp.join(out_dir, f'step{step:04d}_{interaction_type_string}_human_gt'))
    save_pkl({'transl': human_transl_steps,
                'global_orient': human_rot3_steps,
                'pose': human_pose_steps,
                'left_hand_pose': human_lh_pose_steps,
                'right_hand_pose': human_rh_pose_steps,
                'contact': contact_gt}, outpath=osp.join(out_dir, f'step{step:04d}_{interaction_type_string}_human_pred'))
    save_pkl({'transl': partner_transl,
                'global_orient': partner_rot3,
                'pose': partner_pose,
                'left_hand_pose': partner_lh_pose,
                'right_hand_pose': partner_rh_pose,}, outpath=osp.join(out_dir, f'step{step:04d}_{interaction_type_string}_partner'))