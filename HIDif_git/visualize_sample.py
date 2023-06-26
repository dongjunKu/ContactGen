import sys
import os
import os.path as osp
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from egobody import Egobody
from model import load_vposer_v2 as load_vposer
from torch.optim import Adam
from utils import visualize_smpl_params, save_batch_images

# sysargv
dataset_dir = '/workspace/dataset/egobody'
epochs = 100000
start_epoch = 0
batch_size = 5
vposer_ckpt_dir = 'V02_05'
device = 'cuda'

# dataset
egobody_train = Egobody(dataset_dir, mode='train')
dataloader_train = DataLoader(egobody_train, batch_size, shuffle=True)
egobody_test = Egobody(dataset_dir, mode='test')
dataloader_test = DataLoader(egobody_test, 1, shuffle=True)

for step, data in enumerate(dataloader_train):

    human_pose = data['camera_wearer_pose'].to(device)
    partner_pose = data['interactee_pose'].to(device)

    human_batch_images = visualize_smpl_params({'pose_body':human_pose})
    partner_batch_images = visualize_smpl_params({'pose_body':human_pose})

    save_batch_images(human_batch_images[None, ...], outdir='output', outname=f'human_{step}')
    save_batch_images(partner_batch_images[None, ...], outdir='output', outname=f'partner_{step}')

        
