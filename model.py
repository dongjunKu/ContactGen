from tqdm import tqdm
import math
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import smplx_utils
    

class FCBlock(nn.Module):
    def __init__(self, n_dim):
        super(FCBlock, self).__init__()

        self.n_dim = n_dim

        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.acfun = nn.LeakyReLU()

    def forward(self, x0):

        x = self.acfun(self.fc1(x0))
        x = self.acfun(self.fc2(x))
        return x
    

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
    def __init__(self, model, noise_steps=300, beta_start=1e-4, beta_end=0.02):
        super(BaseCDDPM, self).__init__()

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = Parameter(self.prepare_noise_schedule(), requires_grad=False)
        self.alpha = Parameter(1. - self.beta, requires_grad=False)
        self.alpha_hat = Parameter(torch.cumprod(self.alpha, dim=0), requires_grad=False)

        self.model = model
        assert self.model.channels is not None

    def forward(self, x, cond, label, return_x_pred=False, return_t=False):
        batch_size = x.shape[0]
        t = self.sample_timesteps(batch_size).to(x.device)
        x_noisy, noise = self.noise(x, t)
        noise_pred = self.model(x_noisy, cond, label, t)

        return_value = (noise_pred, noise)
        
        if return_x_pred:
            x_pred = self.denoise(x_noisy, noise_pred, t)
            return_value = return_value + (x_pred,)

        if return_t:
            return_value = return_value + (t,)

        return return_value
    
    def denoise(self, x_noisy, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        x = (x_noisy - sqrt_one_minus_alpha_hat * noise) / sqrt_alpha_hat
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

    def sample(self, cond, label, num_samples=1, cfg_scale1=0.5, cfg_scale2=3, callback=None):
        self.model.eval()

        with torch.no_grad():
            x_flows = []
    
            if callback is not None:
                outdicts = []

            x = torch.randn((num_samples, self.model.channels), device=cond.device)
            x_flows.append(x)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (i * torch.ones(num_samples, device=cond.device)).long()
                predicted_noise = self.model(x, cond, label, t)

                if cfg_scale1 > 0:
                    uncond_predicted_noise = self.model(x, None, label, t)
                    if cfg_scale2 > 0:
                        uncond_unlabeled_predicted_noise = self.model(x, None, None, t)
                        uncond_predicted_noise = torch.lerp(uncond_unlabeled_predicted_noise, uncond_predicted_noise, cfg_scale2)
                    predicted_noise = torch.lerp(predicted_noise, uncond_predicted_noise, cfg_scale1) # TODO hard coded
                
                """
                alpha = self.alpha[t][:, None]
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                """

                x_0_pred = self.denoise(x, predicted_noise, t)

                if callback is not None and i > 1:
                    # optimizer
                    grads, outdict = callback(x_0_pred, cond, label, t)
                    x_0_pred = x_0_pred - grads
                    predicted_noise = self.compute_noise(x, x_0_pred, t)

                # DDPM
                """noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - beta / torch.sqrt(1 - alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise"""

                # DDIM
                if i > 1:
                    step = 1
                    alpha_hat_next = self.alpha_hat[t-step][:, None]
                    x = torch.sqrt(alpha_hat_next) * x_0_pred + torch.sqrt(1 - alpha_hat_next) * predicted_noise
                else:
                    x = x_0_pred # torch.sqrt(alpha) * x_0_pred + torch.sqrt(1 - alpha) * predicted_noise

                # save
                if i % (self.noise_steps // 100) == 0 or i == 1:
                    x_flows.append(x_0_pred) # x_flows.append(x)
                    if callback is not None:
                        outdicts.append(outdict)
            
            x_flows = torch.stack(x_flows, dim=1)
        
        self.model.train()

        output = (x, x_flows)
        if callback is not None:
            output += (outdicts,)

        return output


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


class BaseNet(nn.Module):
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

        x_transl, x_rot6, x_pose6, x_lh_pose6, x_rh_pose6 = smplx_utils.split(x)
        x_transl_emb = self.x_transl_emb(x_transl)
        x_rot6_emb = self.x_rot_emb(x_rot6)
        x_pose6_emb = self.x_pose_emb(x_pose6)
        x_lh_pose6_emb = self.x_left_hand_pose_emb(x_lh_pose6)
        x_rh_pose6_emb = self.x_right_hand_pose_emb(x_rh_pose6)
        x_emb = torch.cat([x_transl_emb, x_rot6_emb, x_pose6_emb, x_lh_pose6_emb, x_rh_pose6_emb], axis=-1) # (batch, channels)
        
        if cond is None:
            cond_emb = torch.zeros((x.shape[0], self.ch_hidden), dtype=x.dtype, device=x.device)
        else:
            # for testing
            if cond.shape[0] == 1:
                cond = cond.repeat(x.shape[0], 1)
            cond_transl, cond_rot6, cond_pose6, cond_lh_pose6, cond_rh_pose6 = smplx_utils.split(cond)
            cond_transl_emb = self.cond_transl_emb(cond_transl)
            cond_rot6_emb = self.cond_rot_emb(cond_rot6)
            cond_pose6_emb = self.cond_pose_emb(cond_pose6)
            cond_lh_pose6_emb = self.cond_left_hand_pose_emb(cond_lh_pose6)
            cond_rh_pose6_emb = self.cond_right_hand_pose_emb(cond_rh_pose6)
            cond_emb = torch.cat([cond_transl_emb, cond_rot6_emb, cond_pose6_emb, cond_lh_pose6_emb, cond_rh_pose6_emb], axis=-1)
        
        if label is None:
            label_emb = torch.zeros((x.shape[0], self.ch_hidden), dtype=x.dtype, device=x.device)
        else:
            label_emb = self.label_emb(label.view(-1))
        # for testing
        if label_emb.shape[0] == 1:
            label_emb = label_emb.repeat(x.shape[0], 1)
        
        time_emb = self.time_emb(t)

        h = self.mlp1(x_emb + cond_emb + label_emb + time_emb) # 더하기 때문에 x, cond의 네트워크가 공유되지 않음 => concat으로 바꾸고 공유하자
        h = self.mlp2(h + cond_emb + label_emb + time_emb)
        
        h = self.out(h)

        return h # noise
    

class MyAttentionModule(nn.Module):
    def __init__(self, d_model, nhead: int = 6, dim_feedforward: int = 128, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.x_self_attn = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.cond_self_attn = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.x2cond_cross_attn = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.cond2x_cross_attn = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, x, cond):
        x = self.x_self_attn(x)
        cond = self.cond_self_attn(cond)
        x = self.cond2x_cross_attn(x, cond)
        cond = self.cond2x_cross_attn(cond, x)
        return x, cond
        

class BaseGuideNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.ch_transl = 3
        self.ch_rot = 6
        self.ch_pose = 21 * 6
        self.ch_hand_pose = 15 * 6
        
        self.ch_hidden = 64

        self.pos_emb = nn.Sequential(
            nn.Linear(3, self.ch_hidden),
            nn.LeakyReLU()
        )
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.ch_hidden),
            nn.Linear(self.ch_hidden, self.ch_hidden),
            nn.LeakyReLU()
        )
        self.reg_emb = nn.Embedding(75, self.ch_hidden)
        self.label_emb = nn.Embedding(num_classes, self.ch_hidden)
        
        self.attn1 = MyAttentionModule(d_model=self.ch_hidden, nhead=8, dim_feedforward=2*self.ch_hidden)
        self.attn2 = MyAttentionModule(d_model=self.ch_hidden, nhead=8, dim_feedforward=2*self.ch_hidden)
        self.attn3 = MyAttentionModule(d_model=self.ch_hidden, nhead=8, dim_feedforward=2*self.ch_hidden)

        self.final_signature = nn.Sequential(
            nn.Linear(2*self.ch_hidden, self.ch_hidden),
            nn.ReLU(),
            nn.Linear(self.ch_hidden, 1),
            nn.Sigmoid()
        )

        self.x_segmentation = nn.Sequential(
            nn.Linear(self.ch_hidden, self.ch_hidden),
            nn.ReLU(),
            nn.Linear(self.ch_hidden, 1),
            nn.Sigmoid()
        )

        self.cond_segmentation = nn.Sequential(
            nn.Linear(self.ch_hidden, self.ch_hidden),
            nn.ReLU(),
            nn.Linear(self.ch_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x_smplx, cond_smplx, label, t):

        x_reg = smplx_utils.get_reg_center(x_smplx) # (b, 75, 3)
        x_emb = self.pos_emb(x_reg) # (b, 75, c)
        
        cond_reg = smplx_utils.get_reg_center(cond_smplx) # (b, 75, 3)
        cond_emb = self.pos_emb(cond_reg)
        
        label_emb = self.label_emb(label.view(-1))

        time_emb = self.time_emb(t).unsqueeze(1) # (b, c) -> (b, 1, c)
        label_emb = label_emb.unsqueeze(1) # (b, c) -> (b, 1, c)
        reg_emb = self.reg_emb(torch.tensor([list(range(75))], dtype=torch.long, device=x_reg.device)) # (1, 75, c)

        # jt
        x_emb = x_emb + label_emb + reg_emb + time_emb # (b, 75, c)
        cond_emb = cond_emb + label_emb + reg_emb + time_emb # (b, 75, c)

        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2) # (75, b, c)
        x_emb, cond_emb = self.attn1(x_emb, cond_emb)
        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2)
        
        x_emb = x_emb + label_emb + reg_emb + time_emb
        cond_emb = cond_emb + label_emb + reg_emb + time_emb
        
        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2)
        x_emb, cond_emb = self.attn2(x_emb, cond_emb)
        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2)
        
        x_emb = x_emb + label_emb + reg_emb + time_emb
        cond_emb = cond_emb + label_emb + reg_emb + time_emb
        
        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2)
        x_emb, cond_emb = self.attn3(x_emb, cond_emb)
        x_emb, cond_emb = x_emb.permute(1, 0, 2), cond_emb.permute(1, 0, 2)
        
        x_emb = x_emb + label_emb + reg_emb + time_emb
        cond_emb = cond_emb + label_emb + reg_emb + time_emb

        # seg
        x_seg = self.x_segmentation(x_emb.reshape(-1, self.ch_hidden)).reshape(-1, 75)
        cond_seg = self.cond_segmentation(cond_emb.reshape(-1, self.ch_hidden)).reshape(-1, 75)

        # sig
        sig = torch.cat([x_emb.unsqueeze(2).expand(-1,75,75,-1),
                         cond_emb.unsqueeze(1).expand(x_emb.shape[0],75,75,-1)], dim=-1) # (B, 75, 75, 2c)
        sig = self.final_signature(sig.reshape(-1, 2*self.ch_hidden)).reshape(-1, 75, 75) # (B, 75, 75)

        return sig, x_seg, cond_seg
    
    def sig2sigmark(self, sig, x_seg=None, cond_seg=None, thres=0.5, at_least_one=True):
        """
        sig: torch tensor (B, 75, 75)
        x_seg: torch tensor (B, 75)
        cond_seg: torch tensor (B, 75)
        """

        if x_seg is not None and cond_seg is not None:
            x_seg_mark = x_seg > thres
            cond_seg_mark = cond_seg > thres
            if at_least_one:
                x_seg_max = x_seg.max(dim=1, keepdim=True)[0]
                cond_seg_max = cond_seg.max(dim=1, keepdim=True)[0]
                x_seg_mark_now = (x_seg == x_seg_max)
                cond_seg_mark_now = (cond_seg == cond_seg_max)
                x_seg_mark = x_seg_mark + x_seg_mark_now
                cond_seg_mark = cond_seg_mark + cond_seg_mark_now
            sig = sig * x_seg_mark.unsqueeze(2) * cond_seg_mark.unsqueeze(1)

        sig_mark = torch.zeros_like(sig, dtype=torch.bool)

        if at_least_one:
            sig_max = sig.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] # (B, 1, 1)
            sig_mark_now = (sig == sig_max) # TODO
            sig_mark = sig_mark + sig_mark_now

        sig_mark_now = (sig == sig.max(dim=1, keepdim=True)) * (sig == sig.max(dim=2, keepdim=True)) * (sig > thres)
        sig_mark = sig_mark + sig_mark_now

        return sig_mark
    
    def sigmark2sigidx(self, contact):
        """
        contact: torch bool tensor (B, 75, 75)
        """
        if torch.sum(contact) == 0:
            return []
        else:
            idx = contact.to_sparse().indices().tolist()
            idx = list(zip(idx[0], idx[1], idx[2]))
            return idx