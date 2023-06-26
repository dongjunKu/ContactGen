import os
import os.path as osp
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from body_visualizer.tools.vis_tools import imagearray2file
from body_visualizer.tools.vis_tools import show_image

# sys argv
support_dir = 'body_visualizer/support_data/downloads'

def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


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
    
