import os
import os.path as osp
import pickle
import json
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
import smplx

from params import ParamsAll, ParamsTrain, ParamsSample


# psi code: psi 논문에서 해당 인코딩 방식 사용
class ContinousRotReprDecoder(nn.Module):
    '''
    - this class encodes/decodes rotations with the 6D continuous representation
    - Zhou et al., On the continuity of rotation representations in neural networks
    - also used in the VPoser (see smplx)
    '''

    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot


# psi code
class GeometryTransformer():
    @staticmethod
    def convert_to_6D_rot(xr_aa):
        xr_mat = ContinousRotReprDecoder.aa2matrot(xr_aa) # (b, 3, 3)
        xr_repr =  xr_mat[:,:,:2].reshape([-1,6])
        return xr_repr

    @staticmethod
    def convert_to_3D_rot(xr_repr):
        xr_mat = ContinousRotReprDecoder.decode(xr_repr) # return [:,3,3]
        xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]
        return xr_aa

    @staticmethod
    def recover_global_T(xt_batch, cam_intrisic, max_depth):
        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        # fx_batch = 1000
        # fy_batch = 1000
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        
        z = (xt_batch[:, 2]+1.0)/2.0 * max_depth

        x = xt_batch[:,0] * z / s_ / fx_batch
        y = xt_batch[:,1] * z / s_ / fy_batch
        
        xt_batch_recoverd = torch.stack([x,y,z],dim=-1)

        return xt_batch_recoverd

    @staticmethod
    def normalize_global_T(xt_batch, cam_intrisic, max_depth):
        '''
        according to the camera intrisics and maximal depth,
        normalize the global translate to [-1, 1] for X, Y and Z.
        input: [transl, rotation, local params]
        '''

        fx_batch = cam_intrisic[:,0,0]
        fy_batch = cam_intrisic[:,1,1]
        px_batch = cam_intrisic[:,0,2]
        py_batch = cam_intrisic[:,1,2]
        s_ = 1.0 / torch.max(px_batch, py_batch)
        x = s_* xt_batch[:,0]*fx_batch / (xt_batch[:,2] + 1e-6)
        y = s_* xt_batch[:,1]*fy_batch / (xt_batch[:,2] + 1e-6)

        z = 2.0*xt_batch[:,2] / max_depth - 1.0

        xt_batch_normalized = torch.stack([x,y,z],dim=-1)

        return xt_batch_normalized
    

smplx_train = smplx.create(ParamsAll.smplx_dir,
                            model_type='smplx',
                            gender='neutral',
                            use_pca=False,
                            batch_size=ParamsTrain.batch_size).to(ParamsAll.device)

smplx_test = smplx.create(ParamsAll.smplx_dir,
                            model_type='smplx',
                            gender='neutral',
                            use_pca=False,
                            batch_size=ParamsSample.batch_size).to(ParamsAll.device)

smplx_test_sample = smplx.create(ParamsAll.smplx_dir,
                            model_type='smplx',
                            gender='neutral',
                            use_pca=False,
                            batch_size=ParamsSample.batch_size*ParamsSample.num_samples).to(ParamsAll.device)

class SMPLX_Utils():
    def __init__(self):
        with open(ParamsAll.reg_info_path, 'r') as f:
            self.reg_info = json.load(f)
        self.face = smplx_train.faces # (F, 3)
        self.rid2fids = self.reg_info['rid_to_smplx_fids']
        self.rid2vids = []
        for fids in self.rid2fids:
            vertices = np.unique(self.face[fids]).tolist()
            self.rid2vids.append(vertices)

    def choose_model(self, batch_size):
        if batch_size == ParamsTrain.batch_size:
            self.model = smplx_train
        elif batch_size == ParamsSample.batch_size:
            self.model = smplx_test
        elif batch_size == ParamsSample.batch_size*ParamsSample.num_samples:
            self.model = smplx_test_sample
        else:
            raise ValueError
        
    def smplx(self, betas=None, global_orient=None, body_pose=None,
                left_hand_pose=None, right_hand_pose=None, transl=None,
                expression=None, jaw_pose=None, leye_pose=None, reye_pose=None,
                return_verts=True, return_full_pose=False, pose2rot=True, **kwargs):
        assert body_pose is not None
        self.choose_model(body_pose.shape[0])
        return self.model(betas, global_orient, body_pose,
                            left_hand_pose, right_hand_pose, transl,
                            expression, jaw_pose, leye_pose, reye_pose,
                            return_verts, return_full_pose, pose2rot, **kwargs)
    
    def get_reg_center(self, smplx):
        vertices = smplx.vertices # (B, 10475, 3)
        reg_centers = []
        for vids in self.rid2vids:
            reg_center = torch.mean(vertices[:, vids], dim=1) # (B, 3)
            reg_centers.append(reg_center)
        return torch.stack(reg_centers, dim=1) # (B, 75, 3)
    
    def encode(self, transl, global_orient, body_pose, left_hand_pose, right_hand_pose):
        """
            transl:             (batch_size, 3)
            global_orient:      (batch_size, 3)
            body_pose:          (batch_size, 21*3)
            left_hand_pose:     (batch_size, 15*3)
            right_hand_pose:    (batch_size, 15*3)
            x:                  (batch_size, 3+6+21*6+15*6+15*6)
        """
        x_transl = transl
        x_rot6 = GeometryTransformer.convert_to_6D_rot(global_orient)
        x_pose6 = GeometryTransformer.convert_to_6D_rot(body_pose).reshape(-1, 21*6)
        x_lh_pose6 = GeometryTransformer.convert_to_6D_rot(left_hand_pose).reshape(-1, 15*6)
        x_rh_pose6 = GeometryTransformer.convert_to_6D_rot(right_hand_pose).reshape(-1, 15*6)
        x = torch.cat([x_transl, x_rot6, x_pose6, x_lh_pose6, x_rh_pose6], axis=-1)

        return x
    
    def split(self, x):
        assert x.shape[-1] == 3+6+21*6+15*6+15*6
        temp = 0
        x_transl = x[..., temp:temp+3]
        temp += 3
        x_rot6 = x[..., temp:temp+6]
        temp += 6
        x_pose6 = x[..., temp:temp+21*6]
        temp += 21*6
        x_lh_pose6 = x[..., temp:temp+15*6]
        temp += 15*6
        x_rh_pose6 = x[..., temp:temp+15*6]

        return x_transl, x_rot6, x_pose6, x_lh_pose6, x_rh_pose6
    
    def decode(self, x, return_dict=False):
        x_transl, x_rot6, x_pose6, x_lh_pose6, x_rh_pose6 = self.split(x)
        shapes = x_transl.shape[:-1]

        transl = x_transl
        global_orient = GeometryTransformer.convert_to_3D_rot(x_rot6.reshape(-1, 6)).reshape(shapes+(3,))
        body_pose = GeometryTransformer.convert_to_3D_rot(x_pose6.reshape(-1, 6)).reshape(shapes+(21*3,))
        left_hand_pose = GeometryTransformer.convert_to_3D_rot(x_lh_pose6.reshape(-1, 6)).reshape(shapes+(15*3,))
        right_hand_pose = GeometryTransformer.convert_to_3D_rot(x_rh_pose6.reshape(-1, 6)).reshape(shapes+(15*3,))
    
        if return_dict:
            return {
                'transl': transl,
                'global_orient': global_orient,
                'body_pose': body_pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose
            }
        else:
            return transl, global_orient, body_pose, left_hand_pose, right_hand_pose
    
smplx_utils = SMPLX_Utils()
    
def calc_normal_vectors(vertices, faces):
    """
    vertices: (batch, num_vertices, 3)
    faces: (num_faces, 3)
    vertices_normal: (batch, num_vertices, 3)
    """

    # Create a zero array with the same type and shape as our vertices i.e., per vertex normal
    vertices_normal = torch.zeros_like(vertices)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[:, faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    faces_normal = torch.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    faces_normal = faces_normal / (torch.sum(faces_normal**2, dim=-1, keepdim=True)**0.5) # self.normalize_v3(faces_normal)
    #print('faces_normal:', (torch.sum(faces_normal**2, dim=-1, keepdim=True)**0.5))
    #n = norm(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    vertices_normal[:, faces[:, 0]] = vertices_normal[:, faces[:, 0]] + faces_normal
    vertices_normal[:, faces[:, 1]] = vertices_normal[:, faces[:, 1]] + faces_normal
    vertices_normal[:, faces[:, 2]] = vertices_normal[:, faces[:, 2]] + faces_normal
    #print('vertices_normal:', (torch.sum(vertices_normal**2, dim=-1, keepdim=True)**0.5))
    vertices_normal = vertices_normal / (torch.sum(vertices_normal**2, dim=-1, keepdim=True)**0.5) # self.normalize_v3(vertices_normal)
    # norm(_norm)

    return faces_normal, vertices_normal


def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    elif isinstance(tensor, torch.Tensor): return tensor.detach().cpu().numpy()
    else: return tensor


def save_pkl(body_params, outpath):
    body_param = {key: c2c(body_params[key]) for key in body_params.keys()}
    with open(outpath + '.pkl', 'wb') as f:
        pickle.dump(body_param, f)


def save_batch_pkl(body_params, outdir, outname):
    temp = set([len(body_params[key]) for key in body_params.keys()])
    assert len(temp) == 1
    b = list(temp)[0]
    if b == 1:
        body_param = {key: c2c(body_params[key][0]) for key in body_params.keys()}
        with open(osp.join(outdir, outname + '.pkl'), 'wb') as f:
            pickle.dump(body_param, f)
    else:
        for i in range(b):
            body_param = {key: c2c(body_params[key][i]) for key in body_params.keys()}
            with open(osp.join(outdir, outname + f'_{i}.pkl'), 'wb') as f:
                pickle.dump(body_param, f)


def save_batch_images(batch_images, outdir, outname):
    if len(batch_images.shape) == 5:
        for i in range(batch_images.shape[0]):
            imgs = [Image.fromarray(img) for img in batch_images[i]]
            outpath = osp.join(outdir, outname + f'_{i}.gif')
            imgs[0].save(outpath, save_all=True, append_images=imgs[1:], duration=5, loop=0)
    elif len(batch_images.shape) == 4:
        for i in range(batch_images.shape[0]):
            im = Image.fromarray(batch_images[i])
            im.save(osp.join(outdir, outname + f'_{i}.png'))
    else:
        raise NotImplementedError 
    
