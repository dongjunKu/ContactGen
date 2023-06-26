from tqdm import tqdm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchgeometry as tgm

import smplx


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
    

class ResBlock(nn.Module):
    def __init__(self, n_dim):
        super(ResBlock, self).__init__()

        self.n_dim = n_dim

        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.acfun = nn.LeakyReLU()

    def forward(self, x0):

        x = self.acfun(self.fc1(x0))
        x = self.acfun(self.fc2(x))
        x = x+x0
        return x


class BaseCDDPM(nn.Module):
    def __init__(self, model, optim=None, noise_steps=300, beta_start=1e-4, beta_end=0.02):
        super(BaseCDDPM, self).__init__()

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = Parameter(self.prepare_noise_schedule(), requires_grad=False)
        self.alpha = Parameter(1. - self.beta, requires_grad=False)
        self.alpha_hat = Parameter(torch.cumprod(self.alpha, dim=0), requires_grad=False)

        self.model = model
        self.optim = optim
        assert self.model.channels is not None

    def forward(self, x, cond, label, return_x_pred=False):
        batch_size = x.shape[0]
        t = self.sample_timesteps(batch_size).to(x.device)
        x_noisy, noise = self.noise(x, t)
        noise_pred = self.model(x_noisy, cond, label, t)

        return_value = (noise_pred, noise)
        
        if return_x_pred:
            x_pred = self.denoise(x_noisy, noise_pred, t)
            return_value = return_value + (x_pred,)

        return return_value
    
    def denoise(self, x_noisy, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        x = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat

        if False: # abs(torch.sum(x[0]).item()) > 100:
            print('x_noisy:', x_noisy[0])
            print('noise:', noise[0])
            print('x:', x[0])
            input()
        return x

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise
    
    def compute_noise(self, x_noisy, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        return (x_noisy - sqrt_alpha_hat * x) / sqrt_one_minus_alpha_hat

    def sample_timesteps(self, batch_size):
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))

    def sample(self, cond, label, num_samples=1, cfg_scale1=0.5, cfg_scale2=3, opt_dict=None):
        self.model.eval()

        with torch.no_grad():
            x_flows = []

            x = torch.randn((num_samples, self.model.channels), device=cond.device)
            x_flows.append(x)
            grad_last = 0
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (i * torch.ones(num_samples, device=cond.device)).long()
                predicted_noise = self.model(x, cond, label, t)

                if cfg_scale1 > 0:
                    uncond_predicted_noise = self.model(x, None, label, t)
                    if cfg_scale2 > 0:
                        uncond_unlabeled_predicted_noise = self.model(x, None, None, t)
                        uncond_predicted_noise[:, 9:] = torch.lerp(uncond_unlabeled_predicted_noise, uncond_predicted_noise, cfg_scale2)[:, 9:]
                    predicted_noise[:, 9:] = torch.lerp(predicted_noise, uncond_predicted_noise, cfg_scale1)[:, 9:] # TODO hard coded
                
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                noise = 0.5 * torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                if self.optim is not None and i > 1:
                    # optimizer
                    x_0_pred = self.denoise(x, predicted_noise, t)
                    print('-----------------------------------------------------------------')
                    grads, obj = self.optim.gradient(x_0_pred, opt_dict, True)
                    grad_ct_dist, grad_ct_normal, grad_gravity = grads
                    """
                    grad_ct_dist[grad_ct_dist < torch.median(grad_ct_dist, dim=1, keepdim=True)] = 0
                    grad_ct_normal[grad_ct_normal < torch.median(grad_ct_normal, dim=1, keepdim=True)] = 0
                    grad_gravity[grad_gravity < torch.median(grad_gravity, dim=1, keepdim=True)] = 0
                    """
                    """
                    grad_ct_dist[grad_ct_dist < torch.sort(grad_ct_dist, dim=1, descending=True)[0][:, 10:11]] = 0
                    grad_ct_normal[grad_ct_normal < torch.sort(grad_ct_normal, dim=1, descending=True)[0][:, 10:11]] = 0
                    grad_gravity[grad_gravity < torch.sort(grad_gravity, dim=1, descending=True)[0][:, 10:11]] = 0
                    """
                    # grad_final = 0.2 * (grad_ct_dist + 0.02 * grad_ct_normal + grad_gravity) + 0.95 * grad_last
                    # grad_final = 0.05 * (grad_ct_dist + 0.02 * grad_ct_normal + grad_gravity) + 0.99 * grad_last
                    # grad_final = 0.25 * (grad_ct_dist + 0.01 * grad_ct_normal + grad_gravity) + 0.9 * grad_last
                    grad_final = 0.25 * (grad_ct_dist + grad_gravity) + 0.9 * grad_last
                    grad_last = grad_final
                    x_0_pred = x_0_pred - grad_final
                    # x_0_pred[:,3:] = x_0_pred[:,3:] - grad_final[:,3:]
                    
                    """
                    grad_last = 0
                    for kkk in range(10):
                        grads, obj = self.optim.gradient(x_0_pred, opt_dict, True)
                        grad_ct_dist, grad_ct_normal, grad_gravity = grads
                        # grad_final = 0.2 * grad_ct_dist + 0.2 * grad_ct_normal + 0.2 * grad_gravity + 0.9 * grad_last
                        grad_final = 0.2 * grad_ct_dist + 0.2 * grad_gravity + 0.9 * grad_last
                        x_0_pred = x_0_pred - grad_final
                        grad_last = grad_final
                    """
                    
                    # self.optim.loss(x_0_pred, opt_dict, True)
                    predicted_noise = self.compute_noise(x, x_0_pred, t)
                    # print(obj)

                x = 1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise

                # save
                if i % (self.noise_steps // 100) == 0:
                    x_flows.append(x_0_pred) # x_flows.append(x)
            
            x_flows = torch.stack(x_flows, dim=1)
        
        self.model.train()

        return x, x_flows


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # print("time.shape", time[:, None].shape, "embeddings.shape", embeddings[None, :].shape)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat((torch.zeros_like(embeddings[:, :1]), embeddings), dim=-1)
        # print("embeddings.shape", embeddings.shape)
        return embeddings


class BaseNetX5(nn.Module):
    def __init__(self, ch_hidden=1024, num_layers=2, num_classes=8):
        super().__init__()

        self.ch_transl = 3
        self.ch_rot = 6
        self.ch_pose = 21 * 6
        self.ch_hand_pose = 15 * 6
        
        self.ch_hidden = ch_hidden

        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.ch_hidden),
            nn.Linear(self.ch_hidden, self.ch_hidden),
            nn.LeakyReLU()
        )
        self.label_emb = nn.Embedding(num_classes, self.ch_hidden)
        
        self.x_transl_emb = nn.Linear(self.ch_transl, self.ch_hidden * 2 // 16) # 3 // 16
        self.x_rot_emb = nn.Linear(self.ch_rot, self.ch_hidden * 2 // 16) # 3 // 16
        self.x_pose_emb = nn.Linear(self.ch_pose, self.ch_hidden * 6 // 16) # 4 // 16
        self.x_left_hand_pose_emb = nn.Linear(self.ch_hand_pose, self.ch_hidden * 3 // 16) # 3 // 16
        self.x_right_hand_pose_emb = nn.Linear(self.ch_hand_pose, self.ch_hidden * 3 // 16)

        self.cond_transl_emb = nn.Linear(self.ch_transl, self.ch_hidden * 2 // 16)
        self.cond_rot_emb = nn.Linear(self.ch_rot, self.ch_hidden * 2 // 16)
        self.cond_pose_emb = nn.Linear(self.ch_pose, self.ch_hidden * 6 // 16)
        self.cond_left_hand_pose_emb = nn.Linear(self.ch_hand_pose, self.ch_hidden * 3 // 16)
        self.cond_right_hand_pose_emb = nn.Linear(self.ch_hand_pose, self.ch_hidden * 3 // 16)

        self.num_layers = num_layers
        assert num_layers == 2
        self.mlp1 = ResBlock(self.ch_hidden)
        self.mlp2 = ResBlock(self.ch_hidden)

        self.out = nn.Linear(self.ch_hidden, self.ch_transl + self.ch_rot + self.ch_pose + 2 * self.ch_hand_pose)

        self.channels = self.ch_transl + self.ch_rot + self.ch_pose + 2 * self.ch_hand_pose # ddpm에서 noisy_x 샘플링시 사용


    def forward(self, x, cond, label, t):

        temp = 0
        x_transl_emb = self.x_transl_emb(x[:, temp:temp+self.ch_transl])
        temp += self.ch_transl
        x_rot_emb = self.x_rot_emb(x[:, temp:temp+self.ch_rot])
        temp += self.ch_rot
        x_pose_emb = self.x_pose_emb(x[:, temp:temp+self.ch_pose])
        temp += self.ch_pose
        x_lh_pose_emb = self.x_left_hand_pose_emb(x[:, temp:temp+self.ch_hand_pose])
        temp += self.ch_hand_pose
        x_rh_pose_emb = self.x_right_hand_pose_emb(x[:, temp:temp+self.ch_hand_pose])
        temp += self.ch_hand_pose
        x_emb = torch.cat([x_transl_emb, x_rot_emb, x_pose_emb, x_lh_pose_emb, x_rh_pose_emb], axis=-1)
        
        if cond is None:
            cond_emb = torch.zeros((x.shape[0], self.ch_hidden), dtype=x.dtype, device=x.device)
        else:
            # for testing
            if cond.shape[0] == 1:
                cond = cond.repeat(x.shape[0], 1)
            temp = 0
            cond_transl_emb = self.cond_transl_emb(cond[:, temp:temp+self.ch_transl])
            temp += self.ch_transl
            cond_rot_emb = self.cond_rot_emb(cond[:, temp:temp+self.ch_rot])
            temp += self.ch_rot
            cond_pose_emb = self.cond_pose_emb(cond[:, temp:temp+self.ch_pose])
            temp += self.ch_pose
            cond_lh_pose_emb = self.cond_left_hand_pose_emb(cond[:, temp:temp+self.ch_hand_pose])
            temp += self.ch_hand_pose
            cond_rh_pose_emb = self.cond_right_hand_pose_emb(cond[:, temp:temp+self.ch_hand_pose])
            temp += self.ch_hand_pose
            cond_emb = torch.cat([cond_transl_emb, cond_rot_emb, cond_pose_emb, cond_lh_pose_emb, cond_rh_pose_emb], axis=-1)
        
        if label is None:
            label_emb = torch.zeros((x.shape[0], self.ch_hidden), dtype=x.dtype, device=x.device)
        else:
            label_emb = self.label_emb(label.view(-1))
        # for testing
        if label_emb.shape[0] == 1:
            label_emb = label_emb.repeat(x.shape[0], 1)
        
        time_emb = self.time_emb(t)

        h = self.mlp1(x_emb + cond_emb + label_emb + time_emb)
        h = self.mlp2(h + cond_emb + label_emb + time_emb)
        
        h = self.out(h)

        return h # noise