import numpy as np
from scipy.spatial import cKDTree
import torch
from utils import smplx_utils, calc_normal_vectors


def GMoF(residual, rho):
    squared_res = residual ** 2
    dist = squared_res / (squared_res + rho ** 2)
    return rho ** 2 * dist

def loss_p2p(vtx1, vtx2, n1, n2, rho=0.5): # pytorch3d.loss.chamfer_distance
    """
    vtx1: torch tensor (num_v, 3)
    vtx2: torch tensor (num_v, 3)
    n1: torch tensor (num_v, 3)
    n2: torch tensor (num_v, 3)
    """
    num_vertices_1 = vtx1.size(0)
    num_vertices_2 = vtx2.size(0)
    
    vtx1_expand = vtx1.unsqueeze(1).expand(num_vertices_1, num_vertices_2, 3) # (num_v1, num_v2, 3)
    vtx2_expand = vtx2.unsqueeze(0).expand(num_vertices_1, num_vertices_2, 3)
    
    distances = torch.sum((vtx1_expand - vtx2_expand)**2, dim=2)**0.5 # (num_v1, num_v2)
    min_distances_1, indices1 = torch.min(distances, dim=1) # (num_v1,)
    min_distances_2, indices2 = torch.min(distances, dim=0) # (num_v2,)
    
    n1_expand = n1.unsqueeze(1).expand(num_vertices_1, num_vertices_2, 3) # (num_v1, num_v2, 3)
    n2_expand = n2.unsqueeze(0).expand(num_vertices_1, num_vertices_2, 3)
    
    n_distance = torch.sum((n1_expand + n2_expand)**2, dim=2)**0.5 # torch.sum(n1_expand * n2_expand, dim=2) # (num_v1, num_v2)
    n_distance_1 = n_distance[:, indices1] # (num_v1,)
    n_distance_2 = n_distance[indices2, :] # (num_v2,)
    
    # chamfer_distance = (GMoF(torch.mean(min_distances_1), rho) + GMoF(torch.mean(min_distances_2), rho)) / 2
    chamfer_distance = (torch.mean(GMoF(min_distances_1, rho)) + torch.mean(GMoF(min_distances_2, rho))) / 2
    cos_sim = (torch.mean((1 - GMoF(min_distances_1, rho) / rho**2) * n_distance_1) + torch.mean((1 - GMoF(min_distances_2, rho) / rho**2) * n_distance_2)) / 2

    return chamfer_distance + 0.003 * cos_sim

def loss_contact(smplx1, smplx2, sigidx, progress=None, rho_start=1.0, rho_end=0.1):
    """
    smplx1: smplx
    smplx2: smplx
    contact: (B, 75, 75)
    progress: 0 ~ 1.0, 0 = 0%, 1.0 = 100%
    """
    vertices1 = smplx1.vertices # (B, 10475, 3)
    vertices2 = smplx2.vertices.expand_as(vertices1)

    faces = torch.tensor(smplx_utils.face.astype(np.int32), dtype=torch.long, device=vertices1.device)

    _, normal1 = calc_normal_vectors(vertices1, faces)
    _, normal2 = calc_normal_vectors(vertices2, faces)

    loss = 0
    for b, r1, r2 in sigidx:
        vertices1_now = vertices1[b, smplx_utils.rid2vids[r1]] # (num_v, 3)
        vertices2_now = vertices2[b, smplx_utils.rid2vids[r2]] # (num_v, 3)
        normal1_now = normal1[b, smplx_utils.rid2vids[r1]]
        normal2_now = normal2[b, smplx_utils.rid2vids[r2]]
        if progress is None:
            loss = loss + loss_p2p(vertices1_now, vertices2_now, normal1_now, normal2_now) # / count[b]
        else:
            rho = (1 - progress) * rho_start + progress * rho_end
            loss = loss + loss_p2p(vertices1_now, vertices2_now, normal1_now, normal2_now, rho) # / count[b]

    return loss

def loss_floor(smplx):
    """
    smplx: smplx
    """
    vtx = smplx.vertices # (B, 10475, 3)
    loss = torch.sum(torch.min((vtx[:, :, 2] + 0.9)**2, dim=1)[0])
    return loss