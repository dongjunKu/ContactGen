import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import numpy as np
from scipy.spatial import cKDTree

from model import GeometryTransformer

class BaseOptimizer():

    def __init__(self, model_path, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.smplx_b = smplx.create(model_path,
                                    model_type='smplx',
                                    gender='neutral',
                                    use_pca=False,
                                    batch_size=self.batch_size).to(self.device)
        self.smplx = smplx.create(model_path,
                                  model_type='smplx',
                                  gender='neutral',
                                  use_pca=False,
                                  batch_size=1).to(self.device) # 배치 수가 달라서 2개 만들ㅇ
        

        self.smplx_faces = torch.tensor(self.smplx.faces.astype(np.int32), dtype=torch.long).to(self.device)
        
        self.ch_transl = 3
        self.ch_rot6 = 6
        self.ch_pose6 = 21 * 6
        self.ch_hand_pose6 = 15 * 6

    def loss(self, x, data, vervose=False):
        loss = 0

        temp = 0
        x_transl = x[:, temp:temp+self.ch_transl]
        temp += self.ch_transl
        x_rot6 = x[:, temp:temp+self.ch_rot6]
        temp += self.ch_rot6
        x_pose6 = x[:, temp:temp+self.ch_pose6]
        temp += self.ch_pose6
        x_lh_pose6 = x[:, temp:temp+self.ch_hand_pose6]
        temp += self.ch_hand_pose6
        x_rh_pose6 = x[:, temp:temp+self.ch_hand_pose6]

        x_param = {
            'transl': x_transl,
            'global_orient': GeometryTransformer.convert_to_3D_rot(x_rot6.reshape(-1, 6)).reshape(-1, 3),
            'body_pose': GeometryTransformer.convert_to_3D_rot(x_pose6.reshape(-1, 6)).reshape(-1, 21*3),
            'left_hand_pose': GeometryTransformer.convert_to_3D_rot(x_lh_pose6.reshape(-1, 6)).reshape(-1, 15*3),
            'right_hand_pose': GeometryTransformer.convert_to_3D_rot(x_rh_pose6.reshape(-1, 6)).reshape(-1, 15*3)
        }

        x_vtx = self.smplx_b(**x_param).vertices
        partner_vtx = self.smplx(**data['partner_param']).vertices

        contact = data['contact'].clone() # (B, 2, num_vtx) clone 안붙이면 원본 내용물이 바뀜

        assert partner_vtx.shape[0] == 1 and contact.shape[0] == 1

        # loss = 0

        # contact distance loss
        contact = contact[:, 0] # 1to2, (B, num_vtx)
        mask = contact < 10475 # (1, 10475)
        contact[mask == False] = 0 # 10475 -> 0, out of idx 방지 위해 임시로 0으로 바꿔둠. clone 안붙이면 원본 내용물이 바뀜
        diff = x_vtx - partner_vtx[:,contact.squeeze()] # (5, 10475, 3)
        # diff = x_vtx[:, 4730] - partner_vtx[:, 4730]
        diff = diff[:, mask.squeeze()] # (?, 3)
        # return torch.mean(torch.sum(diff**2, dim=-1)**0.5)
        contact_dist_loss = torch.mean(diff**2)
        # loss = loss + 10 * contact_dist_loss

        # contact normal loss
        _, x_vtx_normal = self.calc_normal_vectors(x_vtx, self.smplx_faces)
        _, partner_vtx_normal = self.calc_normal_vectors(partner_vtx, self.smplx_faces)
        x_vtx_contact = x_vtx[:, mask.squeeze()]
        partner_vtx_contact = partner_vtx[:, contact.squeeze()][:, mask.squeeze()]
        x_vtx_contact_normal = x_vtx_normal[:, mask.squeeze()]
        partner_vtx_contact_normal = partner_vtx_normal[:, contact.squeeze()][:, mask.squeeze()]
        # assume batch_size = 1
        dist, idx = cKDTree(partner_vtx_contact.squeeze().cpu().detach().numpy()).query(x_vtx_contact.squeeze().cpu().detach().numpy()) # idx: (batch_size, num_v)
        mask = []
        for i in range(len(idx)):
            idx_now = np.array(idx[i])
            dist_now = np.array(dist[i])
            mask_now = [False] * len(idx_now)
            for v in np.unique(idx_now):
                dist_now[idx_now != v] = np.inf
                mask_now[np.argmin(dist_now)] = True
            mask.append(mask_now)
        mask = torch.tensor(mask, device=partner_vtx_contact_normal.device)

        inner = torch.sum(partner_vtx_contact_normal * torch.gather(x_vtx_contact_normal, 1, torch.tensor(idx, device=partner_vtx_contact_normal.device).unsqueeze(-1).repeat(1, 1, 3)), dim=-1)[mask] # (B, num_vtx)
        contact_normal_loss = torch.mean(inner)

        # plane loss
        # print(torch.min(partner_vtx[0, :, 0]), torch.min(partner_vtx[0, :, 1]), torch.min(partner_vtx[0, :, 2]))
        gravity_loss = torch.mean(torch.min((x_vtx[:, :, 2] + 0.9)**2, dim=1)[0])

        if vervose:
            print('contact_dist_loss:', contact_dist_loss, 'contact_normal_loss:', contact_normal_loss, 'gravity_loss:', gravity_loss)

        return [contact_dist_loss, contact_normal_loss, gravity_loss]


    def gradient(self, x, data, vervose=False):
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            objs = self.loss(x_in, data, vervose)
            grads = []
            for obj in objs:
                grad = torch.autograd.grad(obj, x_in, retain_graph=True)[0]
                grads.append(grad)
            ## clip gradient by value
            # grad = torch.clip(grad, **self.clip_grad_by_value)
            ## TODO clip gradient by norm

            """
            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')
            """

            # print('max_grad:', torch.max(torch.abs(grads[1])))

            return grads, obj
        
    def calc_normal_vectors(self, vertices, faces):

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
    """
    def normalize_v3(self, arr):
        # Normalize a numpy array of 3 component vectors shape=(n,3)
        lens = torch.sqrt(arr[:, :, 0]**2 + arr[:, :, 1]**2 + arr[:, :, 2]**2)
        arr[:, :, 0] = arr[:, :, 0] / lens
        arr[:, :, 1] = arr[:, :, 1] / lens
        arr[:, :, 2] = arr[:, :, 2] / lens
        return arr
    """