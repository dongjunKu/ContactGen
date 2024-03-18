import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import numpy as np
from scipy.spatial import cKDTree

from params import ParamsDiffusion
from loss import loss_contact, loss_floor
from utils import smplx_utils


def callback_factory(optimizer, guidenet):

    grad_last = 0

    def callback(x, cond, label, t):
        nonlocal grad_last
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            x_params = smplx_utils.decode(x, return_dict=True)
            cond_params = smplx_utils.decode(cond, return_dict=True)

            x_smplx = smplx_utils.smplx(**x_params)
            cond_smplx = smplx_utils.smplx(**cond_params)

            signature_pred, x_segmentation_pred, cond_segmentation_pred = guidenet(x_smplx, cond_smplx, label, t)
            sigidx = guidenet.sigmark2sigidx(guidenet.sig2sigmark(signature_pred, x_segmentation_pred, cond_segmentation_pred))

            l_cnt = loss_contact(x_smplx, cond_smplx, sigidx) #, t[0].item() / )
            # l_cls = loss_collision(x_smplx, cond_smplx, point2plane=True)
            # print('l_cls:', torch.sum(l_cls))
            l_flr = loss_floor(x_smplx)

            # loss_total = 0.25 * l_cnt + l_cls + 0.25 * l_flr # TODO 계수 추가

            grad_cnt = torch.autograd.grad(l_cnt, x, retain_graph=True)[0] # TODO
            # grad_col = torch.autograd.grad(l_cls, x, retain_graph=True)[0] # loss에 오류있음...
            grad_flr = torch.autograd.grad(l_flr, x, retain_graph=True)[0]

            grad = torch.zeros_like(grad_cnt)
            
            # with momentum
            # grad[:,:9] = grad[:,:9] + 0.000001 * grad_cnt[:,:9]
            # grad[:,9:] = grad[:,9:] + 0.4 * grad_cnt[:,9:]
            # grad[:,:3] = grad[:,:3] + 0.1 * grad_flr[:,:3]
            # grad[:,:9] = grad[:,:9] + 0.000001 * grad_col[:,:9]
            # grad[:,9:] = grad[:,9:] + 0.2 * grad_col[:,9:]
            # grad = grad + 0.7 * grad_last
            # grad_last = grad
            
            # without momentum
            grad[:,:3] = grad[:,:3] + 0.2 * grad_flr[:,:3]
            grad[:,:9] = grad[:,:9] + 0.0001 * grad_cnt[:,:9]
            grad[:,9:] = grad[:,9:] + 1.0 * grad_cnt[:,9:]
            # grad[:,:9] = grad[:,:9] + 0.0001 * grad_col[:,:9] # no momentum
            # grad[:,9:] = grad[:,9:] + 5.0 * grad_col[:,9:]

            outdict = {
                'sigidx': sigidx
            }

            return grad, outdict
    
    return callback