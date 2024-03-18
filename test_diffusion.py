import sys
import os
import os.path as osp
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ci3d import CI3D
from model import BaseCDDPM, BaseNet
from params import ParamsSampleDiffusion
from utils import save_pkl, smplx_utils

max_sample_per_label = 10 # s10

# history
history = [max_sample_per_label] * ParamsSampleDiffusion.num_label

# dataset
dataset_test = CI3D(ParamsSampleDiffusion.dataset_dir, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=ParamsSampleDiffusion.batch_size, shuffle=True)
# _, dataloader_test = build_dataloaders(ParamsSampleDiffusion.dataset_dir, ParamsSampleDiffusion.batch_size, num_workers=1)

# models
basenet = BaseNet().to(ParamsSampleDiffusion.device) # ci3d_X5_1024
basecddpm = BaseCDDPM(model=basenet, noise_steps=ParamsSampleDiffusion.noise_steps, beta_start=ParamsSampleDiffusion.beta_start, beta_end=ParamsSampleDiffusion.beta_end).to(ParamsSampleDiffusion.device)

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

for step, data in enumerate(dataloader_test):

    human_transl_gt = data['camera_wearer_transl'].to(ParamsSampleDiffusion.device)
    human_rot3_gt = data['camera_wearer_global_orient'].to(ParamsSampleDiffusion.device)
    human_pose_gt = data['camera_wearer_pose'].to(ParamsSampleDiffusion.device)
    human_lh_pose_gt = data['camera_wearer_left_hand_pose'].to(ParamsSampleDiffusion.device)
    human_rh_pose_gt = data['camera_wearer_right_hand_pose'].to(ParamsSampleDiffusion.device)
    partner_transl = data['interactee_transl'].to(ParamsSampleDiffusion.device)
    partner_rot3 = data['interactee_global_orient'].to(ParamsSampleDiffusion.device)
    partner_pose = data['interactee_pose'].to(ParamsSampleDiffusion.device)
    partner_lh_pose = data['interactee_left_hand_pose'].to(ParamsSampleDiffusion.device)
    partner_rh_pose = data['interactee_right_hand_pose'].to(ParamsSampleDiffusion.device)
    interaction_type = data['interaction_type'].to(ParamsSampleDiffusion.device)

    label = interaction_type.cpu().item()
    if history[label] == 0:
        continue
    else:
        history[label] = history[label] - 1

    partner_feature = smplx_utils.encode(partner_transl, partner_rot3, partner_pose, partner_lh_pose, partner_rh_pose)

    human_feature_pred, human_feature_steps = basecddpm.sample(partner_feature, interaction_type, ParamsSampleDiffusion.num_samples, ParamsSampleDiffusion.cfg_scale1, ParamsSampleDiffusion.cfg_scale2)

    human_transl_pred, human_rot3_pred, human_pose_pred, human_lh_pose_pred, human_rh_pose_pred = smplx_utils.decode(human_feature_pred)
    human_transl_steps, human_rot3_steps, human_pose_steps, human_lh_pose_steps, human_rh_pose_steps = smplx_utils.decode(human_feature_steps)
    interaction_type_string = dataset_test.interaction_types[interaction_type.item()]

    if not osp.exists(ParamsSampleDiffusion.out_dir):
        os.mkdir(ParamsSampleDiffusion.out_dir)
    save_pkl({'transl': human_transl_gt,
                'global_orient': human_rot3_gt,
                'pose': human_pose_gt,
                'left_hand_pose': human_lh_pose_gt,
                'right_hand_pose': human_rh_pose_gt}, outpath=osp.join(ParamsSampleDiffusion.out_dir, f'step{step:04d}_{interaction_type_string}_human_gt'))
    save_pkl({'transl': human_transl_steps,
                'global_orient': human_rot3_steps,
                'pose': human_pose_steps,
                'left_hand_pose': human_lh_pose_steps,
                'right_hand_pose': human_rh_pose_steps}, outpath=osp.join(ParamsSampleDiffusion.out_dir, f'step{step:04d}_{interaction_type_string}_human_pred'))
    save_pkl({'transl': partner_transl,
                'global_orient': partner_rot3,
                'pose': partner_pose,
                'left_hand_pose': partner_lh_pose,
                'right_hand_pose': partner_rh_pose,}, outpath=osp.join(ParamsSampleDiffusion.out_dir, f'step{step:04d}_{interaction_type_string}_partner'))