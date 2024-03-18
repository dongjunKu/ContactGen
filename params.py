class ParamsAll():
    dataset_dir = 'dataset'
    smplx_dir = 'body_models'
    reg_info_path = 'dataset/contact_regions.json'
    num_label = 8

    name = 'ci3d'
    device = 'cuda'


class ParamsTrain(ParamsAll):
    max_epochs = 100000
    batch_size = 32
    save_epoch = 10


class ParamsSample(ParamsAll):
    batch_size = 1
    num_samples = 5


class ParamsDiffusion(ParamsAll):
    name = 'diffusion'
    ckpt_dir = f'checkpoint_{name}_{ParamsAll.name}'
    noise_steps = 1000
    beta_start = 5e-6
    beta_end = 5e-3
    cfg_scale1 = 0.8 # 0.5
    cfg_scale2 = 2 # 3
    

class ParamsGuideNet(ParamsAll):
    name = 'guidenet'
    ckpt_dir = f'checkpoint_{name}_{ParamsAll.name}'


class ParamsTrainDiffusion(ParamsTrain, ParamsDiffusion):
    learning_rate = 1e-4


class ParamsTrainGuideNet(ParamsTrain, ParamsGuideNet):
    w_diffusion = True
    learning_rate = 1e-4


class ParamsSampleDiffusion(ParamsSample, ParamsDiffusion):
    load_epoch = 1000
    out_dir = f'output_{ParamsDiffusion.name}_epoch{load_epoch}_{ParamsAll.name}'


class ParamsSampleGuideNet(ParamsSample, ParamsGuideNet):
    load_epoch = 200
    w_diffusion = True
    out_dir = f'output_{ParamsGuideNet.name}_epoch{load_epoch}_{ParamsAll.name}'