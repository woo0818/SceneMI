import copy
import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
#from eval import eval_humanml, eval_humanact12_uestc
from data_loaders.get_data import get_dataset_loader
from torch.cuda import amp
from utils.parser_util import TrainingOptions
from utils.model_util import load_model_wo_clip
from torch import nn
from diffusion.nn import mean_flat, sum_flat

import wandb
from utils.editing_util import get_keyframes_mask
from utils.utils_transform import *

import pdb

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(self, args: TrainingOptions, model: nn.Module,
                 data, val_data, writer, eval_only = False):
        self.args = args
        self.dataset = args.dataset
        self.model = model
        self.model_avg = None
        self.writer = writer

        self.keyframe_strategy = args.keyframe_strategy
        self.keyframe_interval = args.keyframe_interval
        self.p1 = args.p1
        self.p2 = args.p2

        print(self.keyframe_strategy, self.keyframe_interval)

        if args.avg_model_beta > 0:
            self.model_avg = copy.deepcopy(self.model)

        # self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.val_data = val_data
        self.eval_only = eval_only
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        #self.log_interval = args.log_interval
        self.log_interval = 500
        #pdb.set_trace()
        #self.save_interval = args.save_interval
        self.save_interval = 20000
        self.val_interval = 10000
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = args.use_fp16  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size  # * dist.get_world_size()
        self.num_steps = args.num_steps
        if not self.eval_only:
            self.num_epochs = self.num_steps // len(self.data) + 1

        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        if self.use_fp16:
            # using the new mixed precision trainer
            self.scaler = amp.GradScaler()
            # don't use in fp16
            self.mp_trainer = None
        else:
            self.mp_trainer = MixedPrecisionTrainer(
                model=self.model,
                use_fp16=self.use_fp16,
                fp16_scale_growth=self.fp16_scale_growth,
            )

        self.opt = AdamW(
            # with amp, we don't need to use the mp_trainer's master_params
            (self.model.parameters()
             if self.use_fp16 else self.mp_trainer.master_params),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, args.adam_beta2),
        )

        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = torch.device("cpu")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        #self.schedule_sampler_type = 'uniform'
        #self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        if args.dataset in ['kit', 'humanml'] and args.eval_during_training:
            raise NotImplementedError()
            mm_num_samples = 0  # mm is super slow hence we won't run it during training
            mm_num_repeats = 0  # mm is super slow hence we won't run it during training
            gen_loader = get_dataset_loader(name=args.dataset,
                                            batch_size=args.eval_batch_size,
                                            num_frames=None,
                                            split=args.eval_split,
                                            hml_mode='eval')

            self.eval_gt_data = get_dataset_loader(
                name=args.dataset,
                batch_size=args.eval_batch_size,
                num_frames=None,
                split=args.eval_split,
                hml_mode='gt')
            self.eval_wrapper = EvaluatorMDMWrapper(args.dataset,
                                                    dist_util.dev())
            self.eval_data = {
                'test':
                lambda: eval_humanml.get_mdm_loader(
                    model,
                    diffusion,
                    args.eval_batch_size,
                    gen_loader,
                    mm_num_samples,
                    mm_num_repeats,
                    gen_loader.dataset.opt.max_motion_length,
                    args.eval_num_samples,
                    scale=1.,
                )
            }
        self.use_ddp = False
        #self.ddp_model = self.model

        self.l2_loss = lambda a, b: (
            a - b
        )**2


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(
                resume_checkpoint)
            logger.log(
                f"loading model from checkpoint: {resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                resume_checkpoint, map_location=dist_util.dev())

            if 'model_avg' in state_dict:
                print('loading both model and model_avg')
                state_dict, state_dict_avg = state_dict['model'], state_dict[
                    'model_avg']
                load_model_wo_clip(self.model, state_dict)
                load_model_wo_clip(self.model_avg, state_dict_avg)
            else:
                load_model_wo_clip(self.model, state_dict)
                if self.args.avg_model_beta > 0:
                    # in case we load from a legacy checkpoint, just copy the model
                    print('loading model_avg from model')
                    self.model_avg.load_state_dict(self.model.state_dict())


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint),
                                 f"opt{self.resume_step:09}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(
                f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev())

            if self.use_fp16:
                if 'scaler' not in state_dict:
                    print("scaler state not found ... not loading it.")
                else:
                    # load grad scaler state
                    self.scaler.load_state_dict(state_dict['scaler'])
                    # for the rest
                    state_dict = state_dict['opt']

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])
            # preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd


    def run_loop(self):
        print('train steps:', self.num_steps)
        for epoch in range(self.num_epochs):
            print(f'Starting epoch {epoch}')
            for motion, cond in tqdm(self.data):
                
                if not (not self.lr_anneal_steps or
                        self.step + self.resume_step < self.lr_anneal_steps):
                    break

                motion = motion.to(self.device)
                cond['y'] = {
                    key: val.to(self.device) if torch.is_tensor(val) else val
                    for key, val in cond['y'].items()
                }

                cond['obs_mask'], cond['bps_sbj_mask'] = get_keyframes_mask(motion, cond, edit_mode=self.keyframe_strategy, 
                                                                                        interval=self.keyframe_interval,
                                                                                        p1=self.p1, p2=self.p2)
                self.run_step(motion, cond)

                if self.step % self.log_interval == 0:
                    #logger.dumpkvs()
                    for k, v in logger.get_current().name2val.items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(
                                self.step + self.resume_step, v))
                            self.writer.add_scalar("loss", v, self.step + self.resume_step)

                if self.step > 0 and self.step % self.val_interval == 0:
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save()
                    #self.model.eval()
                    #self.evaluate()
                    #self.model.train()

                    # Run for a finite amount of time in integration tests.
                    
                self.step += 1
            if not (not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps):
                break
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()


    def evaluate(self):
        #if not self.args.eval_during_training:
        #    return
        start_eval = time.time()

        val_losses = []
        inputs = []
        preds = []
        gts = []
        masks = []
        scene_infos = []

        if self.eval_only:
            self.model.eval()
        for motion, cond in self.val_data:
            motion = motion.to(self.device)
            cond['y'] = {
                key: val.to(self.device) if torch.is_tensor(val) else val
                for key, val in cond['y'].items()
            }

            cond['obs_mask'], cond['bps_sbj_mask'] = get_keyframes_mask(motion, cond, edit_mode=self.keyframe_strategy, 
                                                                                    interval=self.keyframe_interval,
                                                                                    p1=self.p1, p2=self.p2)
            val_loss, pred = self.eval_forward_backward(motion, cond)    

            val_losses.append(val_loss.item())
            inputs.append(motion)
            preds.append(pred)
            masks.append(cond['obs_mask'])
            gts.append(cond['y']['gt'])
            scene_infos.append(cond['y']['scene_info'])

            

        print('val_step[{}]: validation_loss[{:0.5f}]'.format(
                                self.step, torch.tensor(val_losses).mean()))
        
        if not self.eval_only:
            self.writer.add_scalar("val_loss", torch.tensor(val_losses).mean(), self.step)

        end_eval = time.time()

        return val_losses, inputs, preds, gts, masks, scene_infos
        
        #print(f'Evaluation time: {round(end_eval-start_eval)/60}min')

    def evaluate_ours(self, motion, bps_sbj, bps_cylinder, key_frames):
        key_frames = key_frames - key_frames[0].item()
        self.model.eval()

        motion = torch.cat([motion['transl'], motion['global_orient_6d'], motion['body_pose_6d']], dim=-1)
        motion, bps_sbj, bps_cylinder = self.val_data.dataset.scene_dataset.transform(motion, bps_sbj.cpu(), bps_cylinder.cpu())

        motion = motion.permute(1,0).unsqueeze(1).unsqueeze(0).float()
        bps_sbj = bps_sbj.permute(1,2,0).unsqueeze(0).float()
        bps_cylinder = bps_cylinder.permute(1,2,0).unsqueeze(0).float()

        max_len = self.val_data.dataset.scene_dataset.max_motion_length
        bs = 1

        motion_key_frames = torch.range(0,20).int().tolist()
        if self.args.keyframe_interval == 15:
            if self.args.num_frames == 120:
                key_frames = [0, 15, 30, 45, 60, 75, 90, 105, 120]
                motion_key_frames = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            elif self.args.num_frames == 60:
                key_frames = [0, 15, 30, 45, 60]
                motion_key_frames = [0, 5, 10, 15, 20]
        if self.args.keyframe_interval == 10:
            if self.args.num_frames == 120:
                key_frames = [0, 9, 18, 27, 36, 45, 54, 63, 72, 84, 96, 108, 120]
                motion_key_frames = [0, 3, 6, 9, 12, 15, 18, 21, 24, 28, 32, 36, 40]

        #pdb.set_trace()

        print(key_frames)
        print(motion_key_frames)

        input_motion = torch.zeros((bs, motion.shape[1], 1, max_len))
        input_bps_sbj = torch.zeros((bs, bps_sbj.shape[1], bps_sbj.shape[2], max_len))

        input_motion[:,:,:,key_frames] = motion[:,:,:,motion_key_frames]
        input_bps_sbj[:,:,:,key_frames] = bps_sbj[:,:,:,motion_key_frames]
        input_motion = input_motion.to(self.device)


        cond = {}
        cond['y'] = {'bps_sbj': input_bps_sbj.to(self.device)}

        cond['obs_mask'], cond['bps_sbj_mask'] = get_keyframes_mask(input_motion, cond, edit_mode=self.keyframe_strategy, 
                                                                                    interval=self.keyframe_interval,
                                                                                    p1=self.p1, p2=self.p2)
        
        
        _, preds = self.eval_forward_backward(input_motion, cond, calculate_loss=False)

        pred_smpl = self.val_data.dataset.scene_dataset.inv_transform(preds[:,:,0].permute(0,2,1).detach().cpu())
        pred_aa = sixd2aa(pred_smpl[:,:,3:],batch=True)

        pred_body_parms = {}
        pred_body_parms['global_orient'] = pred_aa[0, :, 0:3]
        pred_body_parms['body_pose'] = pred_aa[0, :, 3:]
        pred_body_parms['transl'] = pred_smpl[0, :, :3]
        pred_body_parms['betas'] = None
        
        return pred_body_parms

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        # optimizer step
        if self.use_fp16:
            self.scaler.unscale_(self.opt)

            # gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.args.grad_clip)

            grad_norm, param_norm = compute_norms(self.model.parameters())
            logger.logkv('param_norm', param_norm)
            logger.logkv('grad_norm', grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            # FP32, gradient clipping is done in the optimize call
            self.mp_trainer.optimize(self.opt)

        self.update_average_model()
        self._anneal_lr()
        self.log_step()

    def update_average_model(self):
        # update the average model using exponential moving average
        if self.args.avg_model_beta > 0:
            # master params are FP32
            params = self.model.parameters(
            ) if self.use_fp16 else self.mp_trainer.master_params
            for param, avg_param in zip(params, self.model_avg.parameters()):
                # avg = avg + (param - avg) * (1 - alpha)
                # avg = avg + param * (1 - alpha) - (avg - alpha * avg)
                # avg = alpha * avg + param * (1 - alpha)
                avg_param.data.mul_(self.args.avg_model_beta).add_(
                    param.data, alpha=1 - self.args.avg_model_beta)

    def forward_backward(self, batch, cond):
        if self.use_fp16:
            self.opt.zero_grad()
        else:
            self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            #t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())


            # forward pass
            with amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
                model_output = self.model(micro, **micro_cond)
                loss = self.l2_loss(cond['y']['gt'], model_output)
                loss = sum_flat(loss).mean()

            log_loss_dict(loss)

            #pdb.set_trace()

            # backward
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                self.mp_trainer.backward(loss)

    def eval_forward_backward(self, batch, cond, calculate_loss=True):

        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # forward pass
            loss = 0.0
            with amp.autocast(enabled=self.use_fp16, dtype=torch.float16):
                model_output = self.model(micro, **micro_cond)
                if calculate_loss:
                    loss = self.l2_loss(cond['y']['gt'], model_output)
                    loss = sum_flat(loss).mean()
            
            return loss, model_output


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples",
                     (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            if self.use_fp16:
                state_dict = self.model.state_dict()
            else:
                state_dict = self.mp_trainer.master_params_to_state_dict(
                    self.mp_trainer.master_params)
            del_clip(state_dict)

            if self.args.avg_model_beta > 0:
                # save both the model and the average model
                state_dict_avg = self.model_avg.state_dict()
                del_clip(state_dict_avg)
                state_dict = {'model': state_dict, 'model_avg': state_dict_avg}

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        with bf.BlobFile(
                bf.join(self.save_dir,
                        f"opt{(self.step+self.resume_step):09d}.pt"),
                "wb",
        ) as f:
            opt_state = self.opt.state_dict()
            if self.use_fp16:
                # with fp16 we also save the state dict
                opt_state = {
                    'opt': opt_state,
                    'scaler': self.scaler.state_dict(),
                }

            torch.save(opt_state, f)


def compute_norms(params):
    grad_norm = 0.0
    param_norm = 0.0
    for p in params:
        with torch.no_grad():
            param_norm += torch.norm(p, p=2, dtype=torch.float32).item()**2
            if p.grad is not None:
                grad_norm += torch.norm(p.grad, p=2,
                                        dtype=torch.float32).item()**2
    return np.sqrt(grad_norm), np.sqrt(param_norm)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(losses):
    logger.logkv_mean('loss', losses)
