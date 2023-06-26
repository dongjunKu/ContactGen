import os
import os.path as osp
import glob
import json
import pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

import smplx
from model import ContinousRotReprDecoder
# from densepose_methods import dputils

def calc_normal_vectors(vertices, faces):

    vertices_normal = np.zeros_like(vertices)
    tris = vertices[faces]
    faces_normal = np.cross(tris[:, :, 1] - tris[:, :, 0], tris[:, :, 2] - tris[:, :, 0])
    faces_normal = faces_normal / (np.sum(faces_normal**2, axis=-1, keepdims=True)**0.5)
    vertices_normal[faces[:, 0]] = vertices_normal[faces[:, 0]] + faces_normal
    vertices_normal[faces[:, 1]] = vertices_normal[faces[:, 1]] + faces_normal
    vertices_normal[faces[:, 2]] = vertices_normal[faces[:, 2]] + faces_normal
    vertices_normal = vertices_normal / (np.sum(vertices_normal**2, axis=-1, keepdims=True)**0.5)

    return faces_normal, vertices_normal

def check_points_inside(mesh, points, start=0.01, step=0.03):
    # mesh: trimesh.Mesh
    # points: (n, 3) float ndarray
    # return: (n,) bool ndarray
    
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    mesh_tree = cKDTree(mesh.vertices)
    point_tree = cKDTree(points)

    distance = start
    dist_p2m, _ = mesh_tree.query(points, distance_upper_bound=distance)
    candidates = np.isfinite(np.array(dist_p2m)) # (n,) bool
    contained = np.zeros_like(candidates) # (n,) bool
    uncontained = np.zeros_like(candidates) # (n,) bool
    
    while True:
        if not candidates.any():
            break
        points_now = points[candidates]
        contained_now = intersector.contains_points(points_now)
        if not contained_now.any():
            break
        contained_old = contained.copy()
        contained[candidates] = contained_now
        uncontained[candidates] = np.bool_(1 - contained_now)
        contained_now = (np.int32(contained) - np.int32(contained_old)) == 1
        indices = point_tree.query_ball_point(points[contained_now], step)
        indices = list(set(sum(indices, [])))
        candidates[indices] = True # 새로운 애들 추가
        candidates = (np.int32(candidates) - np.int32(contained) - np.int32(uncontained)) == 1 # 이미 포함된 애들은 삭제한다.

    return contained

ci3d_path = '/workspace/dataset/ci3d'
smplx_data_paths = glob.glob(osp.join(ci3d_path, 'chi3d', 'train', '*', 'smplx', '*.json'))
smplx_model_path = "/workspace/dataset/models_smplx_v1_1/models"
gender = "neutral"

# v2i = np.array(dputils.PointID2IUVTable())[:, 0] # (6890,)
body_model = smplx.create(model_path=smplx_model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1).cuda()

# 키재는 코드
if False:
    sample = body_model()
    sample_vertices = sample.vertices.detach().numpy().squeeze()
    v_num = sample_vertices.shape[0]
    height = 0
    for i in tqdm(range(1000000)):
        v1 = np.random.randint(v_num)
        v2 = np.random.randint(v_num)
        v1_v2 = np.sum((sample_vertices[v1,1] - sample_vertices[v2,1])**2)**0.5
        height = max(v1_v2, height)
    
    print(height) # 1.720590353012085 => 아마 172cm
    

for smplx_path in tqdm(smplx_data_paths):
    with open(smplx_path, 'r') as f:
        smplx_data = json.load(f)
    
    temp = smplx_path.split('smplx/')
    info_path = 'smplx/'.join(temp[:-1]) + 'interaction_contact_signature.json'
    with open(info_path, 'r') as f:
        info_data = json.load(f)[temp[-1].replace('.json', '')]
        start_fr, end_fr = info_data['start_fr'], info_data['end_fr']
        num_valid_fr = end_fr - start_fr
    
    nframe = np.array(smplx_data['transl']).shape[1]
    num_c_val_f = 0
    c_val_f = []

    save_data = {
        '1to2': {},
        '2to1': {}
    }

    for idx in tqdm(range(nframe), leave=False):
        smplx1_dict, smplx2_dict = {}, {}
        for key in smplx_data.keys():
            value = np.array(smplx_data[key])
            if value.shape[-2:] == (3, 3):
                value = ContinousRotReprDecoder.matrot2aa(torch.tensor(value[0][idx]).cuda()).float().flatten().unsqueeze(0)
            else:
                value = torch.tensor(value[0][idx]).float().unsqueeze(0).cuda()
            smplx1_dict[key] = value
        for key in smplx_data.keys():
            value = np.array(smplx_data[key])
            if value.shape[-2:] == (3, 3):
                value = ContinousRotReprDecoder.matrot2aa(torch.tensor(value[1][idx]).cuda()).float().flatten().unsqueeze(0)
            else:
                value = torch.tensor(value[1][idx]).float().unsqueeze(0).cuda()
            smplx2_dict[key] = value
        smplx1 = body_model(**smplx1_dict)
        smplx2 = body_model(**smplx2_dict)

        smplx1_vertices = smplx1.vertices.detach().cpu().numpy().squeeze()
        smplx2_vertices = smplx2.vertices.detach().cpu().numpy().squeeze()
        smplx1_mesh = trimesh.Trimesh(smplx1_vertices, body_model.faces)
        smplx2_mesh = trimesh.Trimesh(smplx2_vertices, body_model.faces)
        
        distance_upper_bound = 0.02 # 2cm
        
        dist1to2, vtx1to2 = cKDTree(smplx2_vertices).query(smplx1_vertices)
        dist2to1, vtx2to1 = cKDTree(smplx1_vertices).query(smplx2_vertices)

        dist1to2, vtx1to2 = np.array(dist1to2), np.array(vtx1to2)
        mask1to2_dist = dist1to2 <= distance_upper_bound
        mask1to2_contain = check_points_inside(smplx2_mesh, smplx1_vertices)
        mask1to2_final = (mask1to2_dist + mask1to2_contain) == 0 # 마스킹해야하므로 > 이 아닌 ==
        dist1to2[mask1to2_final] = np.inf
        vtx1to2[mask1to2_final] = 10475
        dist1to2 = list(dist1to2)
        vtx1to2 = list(vtx1to2)

        dist2to1, vtx2to1 = np.array(dist2to1), np.array(vtx2to1)
        mask2to1_dist = dist2to1 <= distance_upper_bound
        mask2to1_contain = check_points_inside(smplx1_mesh, smplx2_vertices)
        mask2to1_final = (mask2to1_dist + mask2to1_contain) == 0 # 마스킹해야하므로 > 이 아닌 ==
        dist2to1[mask2to1_final] = np.inf
        vtx2to1[mask2to1_final] = 10475
        dist2to1 = list(dist2to1)
        vtx2to1 = list(vtx2to1)

        
        if not np.isinf(np.array(dist1to2)).all():
            save_data['1to2'][idx] = {
                                       'dist': dist1to2,
                                       'vtx': vtx1to2
                                     }
        if not np.isinf(np.array(dist2to1)).all():
            save_data['2to1'][idx] = {
                                       'dist': dist2to1,
                                       'vtx': vtx2to1
                                     }
        
    with open(smplx_path.replace('smplx', 'ct_region_1cm').replace('.json', '.pkl'), 'wb') as f:
        pickle.dump(save_data, f)
    
    # print("computed:", num_c_val_f, '/', nframe)
    # print("gt:", num_valid_fr, '/', nframe)