import os
import os.path as osp
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset


class Egobody(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode

        self.data_info = pd.read_csv(osp.join(path, 'data_info_release.csv')).to_numpy()
        self.data_split = pd.read_csv(osp.join(path, 'data_splits.csv')).to_numpy()

        self.path_dict = {}
        
        smplx_camera_wearer_dirs = ['smplx_camera_wearer_train', 'smplx_camera_wearer_val', 'smplx_camera_wearer_test']
        smplx_interactee_dirs = ['smplx_interactee_train', 'smplx_interactee_val', 'smplx_interactee_test']
        for cw_dir, i_dir in zip(smplx_camera_wearer_dirs, smplx_interactee_dirs):
            cw_glob = sorted(glob.glob(osp.join(path, cw_dir, '*')))
            i_glob = sorted(glob.glob(osp.join(path, i_dir, '*')))
            for cw, i in zip(cw_glob, i_glob):
                assert osp.basename(cw) == osp.basename(i)
                self.path_dict[osp.basename(cw)] = (cw, i)
        
        # seminar_j716 scene 데이터만 줍줍
        scene_name = 'seminar_j716'

        self.seminar_j718_rec = self.data_info[self.data_info[:, 0] == scene_name, -1]
        self.seminar_j718_train_records = self.seminar_j718_rec[:-1]
        self.seminar_j718_test_records = self.seminar_j718_rec[-1:]

        self.seminar_j718_train = []
        self.seminar_j718_test = []

        for record in self.seminar_j718_train_records:
            cw_path = osp.join(self.path_dict[record][0], 'body_idx_*', 'results', 'frame_*', '*.pkl')
            cw_paths = sorted(glob.glob(cw_path))
            i_path = osp.join(self.path_dict[record][1], 'body_idx_*', 'results', 'frame_*', '*.pkl')
            i_paths = sorted(glob.glob(i_path))
            assert len(cw_paths) == len(i_paths)
            cal_path = glob.glob(osp.join(self.path, 'calibrations', record, 'cal_trans', 'kinect12_to_world', scene_name + '.json'))[0]
            self.seminar_j718_train += [(cw, i, cal_path) for cw, i in zip(cw_paths, i_paths)]
        
        for record in self.seminar_j718_test_records:
            cw_path = osp.join(self.path_dict[record][0], 'body_idx_*', 'results', 'frame_*', '*.pkl')
            cw_paths = sorted(glob.glob(cw_path))
            i_path = osp.join(self.path_dict[record][1], 'body_idx_*', 'results', 'frame_*', '*.pkl')
            i_paths = sorted(glob.glob(i_path))
            assert len(cw_paths) == len(i_paths)
            cal_path = glob.glob(osp.join(self.path, 'calibrations', record, 'cal_trans', 'kinect12_to_world', scene_name + '.json'))[0]
            self.seminar_j718_test += [(cw, i, cal_path) for cw, i in zip(cw_paths, i_paths)]


    def __len__(self):
        if self.mode == 'train':
            return len(self.seminar_j718_train)
        elif self.mode == 'test':
            return len(self.seminar_j718_test)
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        dataset = self.seminar_j718_train if self.mode == 'train' else self.seminar_j718_test
        cw_path, i_path, cal_path = dataset[idx]

        with open(cw_path, 'rb') as f:
            cw_data = pickle.load(f, encoding='latin1')
        with open(i_path, 'rb') as f:
            i_data = pickle.load(f, encoding='latin1')

        cw_transl = cw_data['transl'].flatten()
        cw_global_orient = cw_data['global_orient'].flatten()
        cw_poz = cw_data['pose_embedding'].flatten()
        cw_pose = cw_data['body_pose'].flatten()

        i_transl = i_data['transl'].flatten()
        i_global_orient = i_data['global_orient'].flatten()
        i_poz = i_data['pose_embedding'].flatten()
        i_pose = i_data['body_pose'].flatten()
            
        item_dict = {
            'camera_wearer_transl': cw_transl,
            'camera_wearer_global_orient': cw_global_orient,
            'camera_wearer_poz': cw_poz,
            'camera_wearer_pose': cw_pose,
            'interactee_transl': i_transl,
            'interactee_global_orient': i_global_orient,
            'interactee_poz': i_poz,
            'interactee_pose': i_pose,
        }

        return item_dict