import smplx
import torch
import pickle
import pyrender
import numpy as np
import trimesh
import json
import tqdm
import sys
import os
import time
from params import ParamsAll
os.environ['KMP_DUPLICATE_LIB_OK']='True'

human_pred_path = sys.argv[1]
partner_path = human_pred_path.replace('human_pred.pkl', 'partner.pkl')
human_gt_path = human_pred_path.replace('human_pred.pkl', 'human_gt.pkl')

model_path = ParamsAll.smplx_dir
contact_regions_template = ParamsAll.reg_info_path
gender = "neutral"
render_diffusion = True
render_gt = False
render_floor = True
num_sample = 5
body_model = smplx.create(model_path=model_path,
                             model_type='smplx',
                             gender=gender,
                             use_pca=False,
                             batch_size=1)

##### color #####
r = np.arange(3, 256, 10, dtype=np.uint8)
g = np.arange(3, 256, 10, dtype=np.uint8)
b = np.arange(3, 256, 10, dtype=np.uint8)
rgb_grid = np.stack(np.meshgrid(r, g, b), axis=-1)
color_space = rgb_grid.reshape((-1, 3)) # (N, 3)

with open(contact_regions_template, 'r') as f:
    contact_regions_template = json.load(f)
rid_to_color = {rid: contact_regions_template['rid_to_color'][rid]
                    for rid in range(len(contact_regions_template['rid_to_color']))}
rid_to_fids = contact_regions_template['rid_to_smplx_fids']

def farthest_point_sampling(points, num_samples):
    """
    points: (N, 3) numpy array
    k: integer
    """
    selected_points = [[0, 0, 255]]
    
    while len(selected_points) < num_samples:
        # Calculate the distance between each point and the selected points
        distances = np.min(np.linalg.norm((points[:, np.newaxis] - selected_points) * np.array([1, 0.8, 1]), axis=2), axis=1)
        # green * 0.8
        
        # Select the point with the maximum distance from the selected points
        farthest_point_index = np.argmax(distances)
        farthest_point = points[farthest_point_index]
        
        # Add the farthest point to the list of selected points
        selected_points.append(farthest_point)
    
    return np.array(selected_points)

def color_mesh_regions(a_trimesh, rid_to_color):
    for rid in rid_to_color:
        a_trimesh.visual.face_colors[rid_to_fids[rid]] = rid_to_color[rid]

##### floor #####
def create_plane_mesh(normal, point):
    # Create a unit vector in the direction of the normal
    normal = np.array(normal) / np.linalg.norm(normal)

    # Define two orthogonal vectors that lie on the plane
    v1, v2, v3 = np.eye(3)
    if np.allclose(normal, v1):
        v1 = np.array([1, 0, 0])
    v1 -= np.dot(v1, normal) * normal
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    # Define vertices for a square plane centered at the origin
    length = 4.0
    vertices = np.array([
        [-length, -length, 0],
        [length, -length, 0],
        [length, length, 0],
        [-length, length, 0],
    ])

    # Translate vertices to the specified point and rotate to align with the normal
    vertices = np.dot(vertices, np.vstack([v1, v2, normal]))
    vertices += np.array(point)

    # Define triangles connecting the vertices
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])

    # Create the trimesh object and return it
    mesh = trimesh.Trimesh(vertices, triangles)

    return mesh

if render_floor:
    plane_tm = create_plane_mesh([0, 0, 1], [0, 0, -0.9])
    plane_mesh = pyrender.Mesh.from_trimesh(plane_tm)

##### partner #####
with open(partner_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    full_poses = torch.tensor(data['pose'], dtype=torch.float32)
    full_rots = torch.tensor(data['global_orient'], dtype=torch.float32)
    #betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
    full_trans = torch.tensor( data['transl'], dtype=torch.float32)
    full_lh_poses = torch.tensor(data['left_hand_pose'], dtype=torch.float32)
    full_rh_poses = torch.tensor(data['right_hand_pose'], dtype=torch.float32)

global_orient = full_rots.reshape(1,-1)
body_pose = full_poses.reshape(1,-1)
transl = full_trans.reshape(1,-1)
left_hand_pose = full_lh_poses.reshape(1,-1)
right_hand_pose = full_rh_poses.reshape(1,-1)
# output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
partner_smplx = body_model(global_orient=global_orient, body_pose=body_pose, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
partner_vtx = partner_smplx.vertices.detach().numpy().squeeze()
partner_tm = trimesh.Trimesh(vertices=partner_vtx, faces=body_model.faces, process=False, vertex_colors=(1.0, 0, 0, 1))

if render_gt:
    ##### human_gt #####
    with open(human_gt_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        full_poses = torch.tensor(data['pose'], dtype=torch.float32)
        full_rots = torch.tensor(data['global_orient'], dtype=torch.float32)
        #betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
        full_trans = torch.tensor( data['transl'], dtype=torch.float32)
        full_lh_poses = torch.tensor(data['left_hand_pose'], dtype=torch.float32)
        full_rh_poses = torch.tensor(data['right_hand_pose'], dtype=torch.float32)

    global_orient = full_rots.reshape(1,-1)
    body_pose = full_poses.reshape(1,-1)
    transl = full_trans.reshape(1,-1)
    left_hand_pose = full_lh_poses.reshape(1,-1)
    right_hand_pose = full_rh_poses.reshape(1,-1)
    # output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
    output = body_model(global_orient=global_orient, body_pose=body_pose, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
    tm = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False, vertex_colors=(0, 1.0, 0, 0.5))
    human_gt_mesh = pyrender.Mesh.from_trimesh(tm)

##### scene #####
scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[1.0, 1.0, 1.0])
cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
cam_pose = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]]
light_pose = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 4],
              [0, 0, 0, 1]]
scene.add(cam, pose=cam_pose)
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)
scene.add(light, pose=light_pose)
# scene.add(light, pose=cam_pose)
if render_floor:
    scene.add(plane_mesh)
if render_gt:
    scene.add(human_gt_mesh)
if not render_diffusion:
    partner_mesh = pyrender.Mesh.from_trimesh(partner_tm)
    scene.add(partner_mesh)

##### human #####
with open(human_pred_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    full_poses = torch.tensor(data['pose'], dtype=torch.float32)
    full_rots = torch.tensor(data['global_orient'], dtype=torch.float32)
    #betas = torch.tensor(data['shape_est_betas'][:10], dtype=torch.float32).reshape(1,10)
    full_trans = torch.tensor( data['transl'], dtype=torch.float32)
    full_lh_poses = torch.tensor(data['left_hand_pose'], dtype=torch.float32)
    full_rh_poses = torch.tensor(data['right_hand_pose'], dtype=torch.float32)
    outdicts = data['outdicts']

if full_poses.ndim == 1:
    global_orient = full_rots.reshape(1,-1)
    body_pose = full_poses.reshape(1,-1)
    transl = full_trans.reshape(1,-1)
    left_hand_pose = full_lh_poses.reshape(1,-1)
    right_hand_pose = full_rh_poses.reshape(1,-1)
    # output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
    output = body_model(global_orient=global_orient, body_pose=body_pose, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
    tm = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False, vertex_colors=(251, 150, 100))
    mesh = pyrender.Mesh.from_trimesh(tm)
    mesh_node = scene.add(mesh, pose=np.eye(4))
    v = pyrender.Viewer(scene)
else:
    ##### preprocess #####

    if render_diffusion:
        
        tms_allframe = []
        partner_tm_allframe = []

        for i in tqdm.tqdm(range(full_poses.shape[1]), desc='preprocessing'): # frame

            signature = outdicts[i]['sigidx'] if i < len(outdicts) else outdicts[-1]['sigidx']
            pallette = farthest_point_sampling(color_space, len(signature))
            pallette = np.concatenate([pallette, 255 * np.ones((len(pallette), 1), dtype=pallette.dtype)], axis=1)

            tms = []

            for j in range(num_sample): # the number of humans
                global_orient = full_rots[j, i].reshape(1,-1)
                body_pose = full_poses[j, i].reshape(1,-1)
                transl = full_trans[j, i].reshape(1,-1)
                left_hand_pose = full_lh_poses[j, i].reshape(1,-1)
                right_hand_pose = full_rh_poses[j, i].reshape(1,-1)
                # output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
                output = body_model(global_orient=global_orient, body_pose=body_pose, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
                output_vertices = output.vertices.detach().numpy().squeeze()
                tm = trimesh.Trimesh(vertices=output_vertices, faces=body_model.faces, process=False, vertex_colors=(251, 150, 100))
                tms.append(tm)

            rid_to_color_human = {(corresp[0], corresp[1]): pallette[i] for i, corresp in enumerate(signature)}
            for bid, rid in rid_to_color_human:
                if bid >= num_sample:
                    continue
                tms[bid].visual.face_colors[rid_to_fids[rid]] = rid_to_color_human[(bid, rid)]
            tms_allframe.append(tms)

            partner_tm_now = partner_tm.copy()
            rid_to_color_partner = {(corresp[0], corresp[2]): pallette[i] for i, corresp in enumerate(signature)}
            for bid, rid in rid_to_color_partner:
                if bid >= num_sample:
                    continue
                partner_tm_now.visual.face_colors[rid_to_fids[rid]] = rid_to_color_partner[(bid, rid)]
            partner_tm_allframe.append(partner_tm_now)
    
    ##### visualize #####
    v = pyrender.Viewer(scene, run_in_thread=True)

    partner_node = None
    mesh_nodes = [None] * num_sample
    line_nodes = [None] * num_sample
    if render_diffusion:
        for i in tqdm.tqdm(range(full_poses.shape[1])): # frame
            v.render_lock.acquire()

            tms = tms_allframe[i]

            for j in range(len(tms)):
                mesh = pyrender.Mesh.from_trimesh(tms[j], smooth=False)
                if mesh_nodes[j] is not None:
                    scene.remove_node(mesh_nodes[j])
                    mesh_nodes[j] = None
                mesh_nodes[j] = scene.add(mesh, pose=np.eye(4))

            partner_tm_now = partner_tm_allframe[i]
            mesh = pyrender.Mesh.from_trimesh(partner_tm_now, smooth=False)
            if partner_node is not None:
                scene.remove_node(partner_node)
                partner_node = None
            partner_node = scene.add(mesh, pose=np.eye(4))
            
            v.render_lock.release()
            input("continue?")

    v.render_lock.acquire()
    for n in mesh_nodes:
        if n is not None:
            scene.remove_node(n)
    for n in line_nodes:
        if n is not None:
            scene.remove_node(n)
    v.render_lock.release()
    
    temp_node = None

    while True:
        for j in range(num_sample):
            v.render_lock.acquire()
            global_orient = full_rots[j, -1].reshape(1,-1)
            body_pose = full_poses[j, -1].reshape(1,-1)
            transl = full_trans[j, -1].reshape(1,-1)
            left_hand_pose = full_lh_poses[j, -1].reshape(1,-1)
            right_hand_pose = full_rh_poses[j, -1].reshape(1,-1)
            # output = body_model(global_orient=global_orient,body_pose=body_pose, betas=betas,transl=transl,return_verts=True)
            output = body_model(global_orient=global_orient, body_pose=body_pose, transl=transl, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, return_verts=True)
            tm = trimesh.Trimesh(vertices=output.vertices.detach().numpy().squeeze(), faces=body_model.faces, process=False, vertex_colors=(251, 150, 100))
            mesh = pyrender.Mesh.from_trimesh(tm)        
            if temp_node is not None:
                scene.remove_node(temp_node)
                temp_node = None
            temp_node = scene.add(mesh, pose=np.eye(4))
            
            v.render_lock.release()
            time.sleep(1)
