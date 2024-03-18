import os
import os.path as osp
import glob
import copy
from tqdm import tqdm
import numpy as np
import json
import pickle
import bisect
import time
import smplx
import random
from scipy.spatial import kdtree
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import Dataset

from utils import ContinousRotReprDecoder


class CI3D(Dataset):
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode
        
        smplx_paths = glob.glob(osp.join(path, 'chi3d', 'train', '*', 'smplx', '*.json'))

        self.interaction_types = ['Grab', 'Handshake', 'Hit', 'HoldingHands', 'Hug', 'Kick', 'Posing', 'Push']
        self.oneside_interaction_types = ['Grab', 'Hit', 'Hug', 'Kick', 'Push']
        self.num_data = {k: 0 for k in self.interaction_types}

        self.train_data = {}
        self.test_data = {}

        with open(osp.join(path, 'chi3d_whoisactor.pkl'), 'rb') as f:
            whoisactor = pickle.load(f)
        
        trainfile_path = osp.join(path, 'chi3d_train.pkl')
        testfile_path = osp.join(path, 'chi3d_test.pkl')

        with open(osp.join(path, 'r_sym_pair.pkl'), 'rb') as f:
            rid_sym_pairs = pickle.load(f)
        self.rid_Tsym_func = np.vectorize(lambda rid: rid_sym_pairs[rid])

        if osp.exists(trainfile_path) and osp.exists(testfile_path):
            with open(trainfile_path, 'rb') as f:
                self.train_data = pickle.load(f)
            with open(testfile_path, 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            for smplx_path in tqdm(smplx_paths):
                with open(smplx_path, 'r') as f:
                    smplx_data = json.load(f)
                num_fr = np.array(smplx_data['transl']).shape[1]

                # valid frame 
                temp = smplx_path.split('smplx/')
                signiture_path = 'smplx/'.join(temp[:-1]) + 'interaction_contact_signature.json'
                with open(signiture_path, 'r') as f:
                    signiture_data = json.load(f)[temp[-1].replace('.json', '')]

                    signature = signiture_data['smplx_signature']['region_id']

                    start_fr, end_fr = signiture_data['start_fr'], signiture_data['end_fr']
                    num_valid_fr = end_fr - start_fr

                    # make label balanced
                    if num_valid_fr < 10:
                        start_fr = max(0, start_fr - 5)
                        end_fr = min(num_fr, end_fr + 5)
                        num_valid_fr = end_fr - start_fr

                # signature
                signature_arr = np.zeros((2, len(smplx_data['transl'][0]), 75, 75))
                for pair in signature:
                    signature_arr[0, :, pair[0], pair[1]] = 1
                    signature_arr[1, :, pair[1], pair[0]] = 1
                smplx_data['signature'] = signature_arr

                # label
                for i, interaction_type in enumerate(self.interaction_types):
                    if interaction_type in smplx_path:
                        smplx_data['label'] = i * np.ones((2, len(smplx_data['transl'][0]), 1))
                        self.num_data[interaction_type] += num_valid_fr

                for key in ['transl', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'label', 'signature']:
                    data_now = np.array(smplx_data[key])[:, start_fr:end_fr]

                    for interaction_type in self.oneside_interaction_types:
                        if interaction_type in smplx_path:
                            for s in ['s02', 's03', 's04']:
                                if s in smplx_path:
                                    break
                            name = smplx_path.split('/')[-1].split('.')[0]

                            # actor, actee
                            if whoisactor[s][name] == 1:
                                data_now = data_now[::-1]
                            break

                    if 's02' in smplx_path: # test: s02
                        if key not in self.test_data.keys():
                            self.test_data[key] = data_now
                        else:
                            self.test_data[key] = np.concatenate([self.test_data[key], data_now], axis=1)
                    else: # train: s03, s04
                        if key not in self.train_data.keys():
                            self.train_data[key] = data_now
                        else:
                            self.train_data[key] = np.concatenate([self.train_data[key], data_now], axis=1)
            
            print('num_data:', self.num_data)

            with open(trainfile_path, 'wb') as f:
                pickle.dump(self.train_data, f)
            with open(testfile_path, 'wb') as f:
                pickle.dump(self.test_data, f)

    
    def weight4balancedsampling(self):
        dataset = self.train_data if self.mode == 'train' else self.test_data
        labels = dataset['label'][0].astype(np.int).flatten()
        uniques, counts = np.unique(labels, return_counts=True)
        weights = np.sum(counts) / counts
        weights = weights[labels]
        return weights
    
    
    def normalize(self, param):
        # param = param.copy()
        param = copy.deepcopy(param)

        param['camera_wearer_transl'] = torch.tensor(param['camera_wearer_transl'] - param['interactee_transl'])
        param['interactee_transl'] = torch.tensor(param['interactee_transl'] - param['interactee_transl'])

        # initial: axis=(0, 1, 0), angle=0
        rot_x = torch.tensor([[1, 0, 0],
                            [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                            [0, np.sin(np.pi/2), np.cos(np.pi/2)]], dtype=torch.float)
        
        rot1 = ContinousRotReprDecoder.aa2matrot(torch.tensor(param['camera_wearer_global_orient']))
        rot2 = ContinousRotReprDecoder.aa2matrot(torch.tensor(param['interactee_global_orient']))

        rot1 = torch.matmul(rot1, torch.inverse(rot_x))
        rot2 = torch.matmul(rot2, torch.inverse(rot_x))

        # delete z-rotation
        rot2 = Rotation.from_matrix(rot2.numpy())
        rot2_euler =  rot2.as_euler('ZXY')
        theta = -rot2_euler[0, 0]
        rot2_euler[0, 0] = rot2_euler[0, 0] + theta
        rot2 = torch.tensor(Rotation.from_euler('ZXY', rot2_euler).as_matrix(), dtype=torch.float)

        rot1 = Rotation.from_matrix(rot1.numpy())
        rot1_euler =  rot1.as_euler('ZXY')
        rot1_euler[0, 0] = rot1_euler[0, 0] + theta
        rot1 = torch.tensor(Rotation.from_euler('ZXY', rot1_euler).as_matrix(), dtype=torch.float)
        
        rot_z = torch.tensor([[np.cos(theta), np.sin(theta), 0],
                            [-np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]], dtype=torch.float)        
        param['camera_wearer_transl'] =  torch.matmul(param['camera_wearer_transl'], rot_z)

        rot1 = torch.matmul(rot1, rot_x)
        rot2 = torch.matmul(rot2, rot_x)
        
        param['camera_wearer_global_orient'] = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1)
        param['interactee_global_orient'] = ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
        
        return param
    
    def augument_Tsym(self, param):
        param = copy.deepcopy(param)

        ################################################
        ################ global orient #################
        ################################################
        # assuming already normalized
        rot_x = torch.tensor([[1, 0, 0],
                            [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                            [0, np.sin(np.pi/2), np.cos(np.pi/2)]], dtype=torch.float) # x 정답! (0, 1, 0) => (0, 0, 1)
        
        rot1 = ContinousRotReprDecoder.aa2matrot(param['camera_wearer_global_orient'])
        rot2 = ContinousRotReprDecoder.aa2matrot(param['interactee_global_orient'])

        rot1 = torch.matmul(rot1, torch.inverse(rot_x))
        rot2 = torch.matmul(rot2, torch.inverse(rot_x))

        # flip on y-axis
        rot2 = Rotation.from_matrix(rot2.numpy())
        rot2_euler =  rot2.as_euler('YZX')
        rot2_euler[0, 0] = -rot2_euler[0, 0]
        rot2_euler[0, 1] = -rot2_euler[0, 1]
        rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)
        rot1 = Rotation.from_matrix(rot1.numpy())
        rot1_euler =  rot1.as_euler('YZX')
        rot1_euler[0, 0] = -rot1_euler[0, 0]
        rot1_euler[0, 1] = -rot1_euler[0, 1]
        rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)

        rot1 = torch.matmul(rot1, rot_x)
        rot2 = torch.matmul(rot2, rot_x)
        
        param['camera_wearer_global_orient'] = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1)
        param['interactee_global_orient'] = ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)

        ################################################
        #################### transl ####################
        ################################################
        param['camera_wearer_transl'] = param['camera_wearer_transl'] * torch.tensor([-1, 1, 1])

        ################################################
        ##################### pose #####################
        ################################################
        # https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
        """
        0: "pelvis", => 얘는 global orient로 빠짐.
        0: "left_hip",
        1: "right_hip",
        2: "spine1",
        3: "left_knee",
        4: "right_knee",
        5: "spine2",
        6: "left_ankle",
        7: "right_ankle",
        8: "spine3",
        9: "left_foot",
        10: "right_foot",
        11: "neck",
        12: "left_collar",
        13: "right_collar",
        14: "head",
        15: "left_shoulder",
        16: "right_shoulder",
        17: "left_elbow",
        18: "right_elbow",
        19: "left_wrist",
        20: "right_wrist",
        21: "jaw"
        """

        sole_pose = [2, 5, 8, 11, 14]
        pair_pose = [(0, 1), (3, 4), (6, 7), (9, 10), (12, 13), (15, 16), (17, 18), (19, 20)]
        for pose in sole_pose:
            rot1 = ContinousRotReprDecoder.aa2matrot(torch.tensor(param['camera_wearer_pose'][3*pose:3*pose+3]))
            rot2 = ContinousRotReprDecoder.aa2matrot(torch.tensor(param['interactee_pose'][3*pose:3*pose+3]))
            
            # flip on z-axis
            rot1 = Rotation.from_matrix(rot1.numpy())
            # print('before:', rot1)
            rot1_euler =  rot1.as_euler('YZX')
            rot1_euler[0, 0] = -rot1_euler[0, 0]
            rot1_euler[0, 1] = -rot1_euler[0, 1]
            rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)
            rot2 = Rotation.from_matrix(rot2.numpy())
            rot2_euler =  rot2.as_euler('YZX')
            rot2_euler[0, 0] = -rot2_euler[0, 0]
            rot2_euler[0, 1] = -rot2_euler[0, 1]
            rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)
            
            param['camera_wearer_pose'][3*pose:3*pose+3] = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1)
            param['interactee_pose'][3*pose:3*pose+3] = ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
            

        for right, left in pair_pose:
            # param1
            rot1, rot2 = torch.tensor(param['camera_wearer_pose'][3*left:3*left+3]).clone(), torch.tensor(param['camera_wearer_pose'][3*right:3*right+3]).clone()
            rot1, rot2 = ContinousRotReprDecoder.aa2matrot(rot1), ContinousRotReprDecoder.aa2matrot(rot2)

            rot1 = Rotation.from_matrix(rot1.numpy())
            rot1_euler =  rot1.as_euler('YZX')
            rot1_euler[0, 0] = -rot1_euler[0, 0]
            rot1_euler[0, 1] = -rot1_euler[0, 1]
            rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)
            rot2 = Rotation.from_matrix(rot2.numpy())
            rot2_euler =  rot2.as_euler('YZX')
            rot2_euler[0, 0] = -rot2_euler[0, 0]
            rot2_euler[0, 1] = -rot2_euler[0, 1]
            rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)

            rot1, rot2 = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1), ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
            param['camera_wearer_pose'][3*left:3*left+3], param['camera_wearer_pose'][3*right:3*right+3] = rot2, rot1
            
            # param2
            rot1, rot2 = torch.tensor(param['interactee_pose'][3*left:3*left+3]).clone(), torch.tensor(param['interactee_pose'][3*right:3*right+3]).clone()
            rot1, rot2 = ContinousRotReprDecoder.aa2matrot(rot1), ContinousRotReprDecoder.aa2matrot(rot2)

            rot1 = Rotation.from_matrix(rot1.numpy())
            rot1_euler =  rot1.as_euler('YZX')
            rot1_euler[0, 0] = -rot1_euler[0, 0]
            rot1_euler[0, 1] = -rot1_euler[0, 1]
            rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)
            rot2 = Rotation.from_matrix(rot2.numpy())
            rot2_euler =  rot2.as_euler('YZX')
            rot2_euler[0, 0] = -rot2_euler[0, 0]
            rot2_euler[0, 1] = -rot2_euler[0, 1]
            rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)

            rot1, rot2 = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1), ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
            param['interactee_pose'][3*left:3*left+3], param['interactee_pose'][3*right:3*right+3] = rot2, rot1

        
        ################################################
        ################## hands pose ##################
        ################################################
        for i in range(15):
            # param1
            rot1, rot2 = torch.tensor(param['camera_wearer_left_hand_pose'][3*i:3*i+3]).clone(), torch.tensor(param['camera_wearer_right_hand_pose'][3*i:3*i+3]).clone()
            rot1, rot2 = ContinousRotReprDecoder.aa2matrot(rot1), ContinousRotReprDecoder.aa2matrot(rot2)

            rot1 = Rotation.from_matrix(rot1.numpy())
            rot1_euler =  rot1.as_euler('YZX')
            rot1_euler[0, 0] = -rot1_euler[0, 0]
            rot1_euler[0, 1] = -rot1_euler[0, 1]
            rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)
            rot2 = Rotation.from_matrix(rot2.numpy())
            rot2_euler =  rot2.as_euler('YZX')
            rot2_euler[0, 0] = -rot2_euler[0, 0]
            rot2_euler[0, 1] = -rot2_euler[0, 1]
            rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)

            rot1, rot2 = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1), ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
            param['camera_wearer_left_hand_pose'][3*i:3*i+3], param['camera_wearer_right_hand_pose'][3*i:3*i+3] = rot2, rot1

            # param2
            rot1, rot2 = torch.tensor(param['interactee_left_hand_pose'][3*i:3*i+3]).clone(), torch.tensor(param['interactee_right_hand_pose'][3*i:3*i+3]).clone()
            rot1, rot2 = ContinousRotReprDecoder.aa2matrot(rot1), ContinousRotReprDecoder.aa2matrot(rot2)

            rot1 = Rotation.from_matrix(rot1.numpy())
            rot1_euler =  rot1.as_euler('YZX')
            rot1_euler[0, 0] = -rot1_euler[0, 0]
            rot1_euler[0, 1] = -rot1_euler[0, 1]
            rot1 = torch.tensor(Rotation.from_euler('YZX', rot1_euler).as_matrix(), dtype=torch.float)
            rot2 = Rotation.from_matrix(rot2.numpy())
            rot2_euler =  rot2.as_euler('YZX')
            rot2_euler[0, 0] = -rot2_euler[0, 0]
            rot2_euler[0, 1] = -rot2_euler[0, 1]
            rot2 = torch.tensor(Rotation.from_euler('YZX', rot2_euler).as_matrix(), dtype=torch.float)

            rot1, rot2 = ContinousRotReprDecoder.matrot2aa(rot1).reshape(-1), ContinousRotReprDecoder.matrot2aa(rot2).reshape(-1)
            param['interactee_left_hand_pose'][3*i:3*i+3], param['interactee_right_hand_pose'][3*i:3*i+3] = rot2, rot1

        
        ################################################
        ################### signature ##################
        ################################################
        signature = param['signature']
        first, second = np.where(signature == 1)
        signature = np.zeros_like(signature)
        first, second = self.rid_Tsym_func(first), self.rid_Tsym_func(second)
        signature[first, second] = 1
        param['signature'] = signature
        
        
        return param
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data['transl'][0])
        elif self.mode == 'test':
            return len(self.test_data['transl'][0])
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        dataset = self.train_data if self.mode == 'train' else self.test_data

        cw_transl = np.array(dataset['transl'][0][idx]).flatten().astype(np.float32)
        cw_global_orient = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['global_orient'][0][idx])).float().numpy().flatten()
        cw_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['body_pose'][0][idx])).float().numpy().flatten()
        cw_lh_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['left_hand_pose'][0][idx])).float().numpy().flatten()
        cw_rh_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['right_hand_pose'][0][idx])).float().numpy().flatten()

        i_transl = np.array(dataset['transl'][1][idx]).flatten().astype(np.float32)
        i_global_orient = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['global_orient'][1][idx])).float().numpy().flatten()
        i_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['body_pose'][1][idx])).float().numpy().flatten()
        i_lh_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['left_hand_pose'][1][idx])).float().numpy().flatten()
        i_rh_pose = ContinousRotReprDecoder.matrot2aa(torch.tensor(dataset['right_hand_pose'][1][idx])).float().numpy().flatten()

        i_type = dataset['label'][0][idx].flatten().astype(np.int)
        signature = dataset['signature'][0, idx].astype(np.float32)
            
        item_dict = {
            'camera_wearer_transl': cw_transl,
            'camera_wearer_global_orient': cw_global_orient,
            'camera_wearer_pose': cw_pose,
            'camera_wearer_left_hand_pose': cw_lh_pose,
            'camera_wearer_right_hand_pose': cw_rh_pose,
            'interactee_transl': i_transl,
            'interactee_global_orient': i_global_orient,
            'interactee_pose': i_pose,
            'interactee_left_hand_pose': i_lh_pose,
            'interactee_right_hand_pose': i_rh_pose,
            'interaction_type': i_type,
            'signature': signature
        }

        """
        camera_wearer_transl (3,)
        camera_wearer_global_orient (3,)
        camera_wearer_pose (63,)
        camera_wearer_left_hand_pose (45,)
        camera_wearer_right_hand_pose (45,)
        interactee_transl (3,)
        interactee_global_orient (3,)
        interactee_pose (63,)
        interactee_left_hand_pose (45,)
        interactee_right_hand_pose (45,)
        interaction_type (1,)
        signature (75, 75)
        """

        item_dict = self.normalize(item_dict)
        if self.mode == 'train' and random.random() > 0.5:
            item_dict = self.augument_Tsym(item_dict)

        return item_dict
    