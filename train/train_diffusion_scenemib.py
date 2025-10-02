# This code is based on https://github.com/openai/guided-diffusion,
# and is used to train a diffusion model on human motion sequences.

import os
import sys
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_diffusion_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_saved_model
from configs import card
from torch.utils.tensorboard import SummaryWriter

def list_of_floats(arg):
    return list(map(float, arg.split(',')))

def main():
    #args = train_args(base_cls=card.motion_smpl_unet_adagn_xl) # Choose the default full motion model from GMD
    args = train_args(base_cls=card.motion_smpl_mdm) # Choose the default full motion model from GMD
    #init_wandb(config=args)
    args.noise = list_of_floats(args.noise)

    if args.keyframe_strategy == "uniform":
        select_info = args.keyframe_interval
    else:
        select_info = [args.p1, args.p2]

    model_info = "diffusion_scenemib"
    
    writer = SummaryWriter(os.path.join("runs/diffusion", model_info))
    args.save_dir = os.path.join("save", model_info)
    pprint(args.__dict__)
    fixseed(args.seed)


    print("model will be saved at {}".format(args.save_dir))

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        scene_enc=args.scene_type,
        noise=args.noise,
        data_rep=args.data_rep,
        trunc_bps=args.trunc_bps,
        light_bps=args.light_bps,
        sub_bps=args.sub_bps,
        beta=args.beta,
        body_abstract=args.body_abstract,
        scene_size=args.scene_size,
        split='train'
    )

    data = get_dataset_loader(data_conf)

    val_data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        scene_enc=args.scene_type,
        noise=args.noise,
        data_rep=args.data_rep,
        trunc_bps=args.trunc_bps,
        light_bps=args.light_bps,
        sub_bps=args.sub_bps,
        beta=args.beta,
        body_abstract=args.body_abstract,
        scene_size=args.scene_size,
        split='val'
    )
    val_data = get_dataset_loader(val_data_conf)

    print("creating model...")
    model, diffusion = create_model_and_diffusion(args, data)
    diffusion.data_inv_transform_fn = data.dataset.scene_dataset.inv_transform_cuda
    #model = create_model(args, data)

    if args.init_model_path is not None:
        ###################################
        # LOADING THE MODEL FROM CHECKPOINT
        print(f"Loading checkpoints from [{args.init_model_path}]...")
        load_saved_model(model, args.init_model_path) # , use_avg_model=args.gen_avg_model)
        niter = int(os.path.basename(args.init_model_path).replace('model', '').replace('.pt', ''))
    else:
        niter = 0
    
    model.to(dist_util.dev())

    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, model, diffusion, data, val_data, niter, writer).run_loop()


if __name__ == "__main__":
    main()