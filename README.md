# ContactGen: Contact-Guided Interactive 3D Human Generation for Partners (AAAI 2024)
[project page](https://dongjunku.github.io/contactgen/)

[paper](https://ojs.aaai.org/index.php/AAAI/article/view/27962)

[arxiv](https://arxiv.org/abs/2401.17212)

## Install

### Docker
```shell
docker pull dongjunku/humaninter:cu11
```

### Project Structure
Please organize your project in the following structure:
```bash
ContactGen
├── body_models
|   ├── smplx
|   |   ├── SMPLX_NEUTRAL.npz
|   |   ├── SMPLX_NEUTRAL.pkl
├── datasets
|   ├── chi3d
|   |   ├── train
|   |   |   ├── s02
|   |   |   |   ├── camera_parameters
|   |   |   |   ├── gpp
|   |   |   |   ├── joints3d_25
|   |   |   |   ├── smplx
|   |   |   |   ├── videos
|   |   |   |   ├── interaction_contact_signature.json
|   |   |   ├── s03
|   |   |   ├── s04
|   ├── chi3d_whoisactor.pkl
|   ├── contact_regions.json
|   ├── r_sym_pair.pkl
├── ci3d.py
├── loss.py
├── model.py
├── optimizer.py
├── params.py
├── sample.py
├── test_diffusion.py
├── test_guidenet.py
├── train_diffusion.py
├── train_guidenet.py
├── utils.py
├── visualize.py
```
You can get CHI3D dataset [here](https://ci3d.imar.ro/chi3d)

You can get SMPL-X [here](https://smpl-x.is.tue.mpg.de/download.php)

You can get ``contact_regions.json`` [here](https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/main/info/contact_regions.json)

## Pretrained Model
The pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1ZqWq_yPoEtig2UvHngqcJfgVyjFvCRfc?usp=drive_link). After downloading, place ``checkpoint_diffusion_ci3d`` and ``checkpoint_guidenet_ci3d`` in the ``ContactGen/``.

## Training
Diffusion module should be trained first
```shell
python train_diffusion.py
```
Next you can train guidenet
```shell
python train_guidenet.py
```
## Sampling
```shell
python sample.py
```
It will generate samples in the ``output_diffusion_epoch1000_ci3d``

## Visualizing
```shell
python visualize.py output_diffusion_epoch1000_ci3d/???_human_pred.pkl
```
You can visualize overall diffusion steps using ``visualize.py``
