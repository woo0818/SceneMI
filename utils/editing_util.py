import torch
import os
import numpy as np
import random
from data_loaders import humanml_utils, amass_utils

def bool_matmul(a, b):
    res = torch.matmul(a.float(), b.float().to(a.device))
    assert (res == res.bool()).all()
    return res.bool()


def joint_to_full_mask_amass(joint_mask, mode='all'):
    # If mode='nemf', choose features corresponding to pos, global_xform, trans, root_orient to be consistent with NeMF
    # But we choose to use everything except velocities and contacts. 609 features / 764 features
    joint_mask = joint_mask.permute(2, 3, 0, 1) # [1, seqlen, bs, 24]
    mask_comp = []
    mask_comp.append(bool_matmul(joint_mask, torch.tensor(amass_utils.MAT_POS)))
    mask_comp.append(bool_matmul(joint_mask, torch.tensor(amass_utils.MAT_ROTMAT)))
    mask_comp.append(bool_matmul(joint_mask, torch.tensor(amass_utils.MAT_ROT)))
    if mode == 'all':
        mask_comp.append(bool_matmul(joint_mask, torch.tensor(amass_utils.MAT_HEIGHT)))
        mask_comp.append(bool_matmul(joint_mask, torch.tensor(amass_utils.MAT_ROT6D)))

    mask = torch.stack(mask_comp, dim=0).any(dim=0) # [1, seqlen, bs, 764]
    return mask.permute(2, 3, 0, 1) # [bs, 764, 1, seqlen]


def joint_to_full_mask(joint_mask, mode='pos_rot_vel'):
    assert mode in ['pos', 'pos_rot', 'pos_rot_vel']
    # joint_mask.shape = [bs, 22, 1, seqlen]
    joint_mask = joint_mask.permute(2, 3, 0, 1) # [1, seqlen, bs, 22]

    mask_comp = []
    mask_comp.append(bool_matmul(joint_mask, torch.tensor(humanml_utils.MAT_POS)))
    mask_comp.append(bool_matmul(joint_mask, torch.tensor(humanml_utils.MAT_CNT)))
    if mode in ['pos_rot', 'pos_rot_vel']:
        mask_comp.append(bool_matmul(joint_mask, torch.tensor(humanml_utils.MAT_ROT)))
    if mode == 'pos_rot_vel':
        mask_comp.append(bool_matmul(joint_mask, torch.tensor(humanml_utils.MAT_VEL)))

    mask = torch.stack(mask_comp, dim=0).any(dim=0) # [1, seqlen, bs, 263]
    return mask.permute(2, 3, 0, 1) # [bs, 263, 1, seqlen]


def get_random_binary_mask(dim1, dim2, n):
    valid_indices = torch.nonzero(torch.ones(dim1, dim2), as_tuple=False)
    flat_indices = np.random.choice(np.arange(dim1*dim2), n, replace=False)
    indices = valid_indices[flat_indices]
    mask = torch.zeros((dim1, dim2), dtype=torch.bool)
    mask[indices[:, 0], indices[:, 1]] = 1
    return mask

def get_keyframes_mask(data, cond, edit_mode='uniform', interval=3, p1=0.1, p2=0.3, given_mask=None, verbose=False):
    obs_mask = torch.zeros_like(data, dtype=bool, device=data.device)
    feature_sbj_mask = torch.zeros_like(cond['y']['bps_sbj'], dtype=bool, device=data.device)

    length = data.shape[-1]


    if given_mask is None:
        if edit_mode == "uniform":
            gt_indices = np.array(range(length)[::interval])
            gt_indices = np.concatenate((gt_indices, np.array([0, length-1])))
            obs_mask[:, :, :, gt_indices] = True  # set keyframes
            feature_sbj_mask[:, :, :, gt_indices] = True

        elif edit_mode == "random":
            #num_keyframes = np.random.randint(length // 5, length // 3)
            num_keyframes = np.random.randint(int(length * p1), int(length * p2))
            gt_indices = np.random.choice(range(length), num_keyframes, replace=False)
            gt_indices = np.concatenate((gt_indices, np.array([0, length-1])))
            
            obs_mask[:, :, :, gt_indices] = True  # set keyframes
            feature_sbj_mask[:, :, :, gt_indices] = True

    else:
        given_mask = given_mask + [0, length-1]
        gt_indices = np.array(given_mask)

        obs_mask[:, :, :, gt_indices] = True  # set keyframes
        feature_sbj_mask[:, :, :, gt_indices] = True


    return obs_mask, feature_sbj_mask



def relative_to_global(data):
    from data_loaders.humanml.scripts.motion_process import recover_root_ang_pos
    output = data.clone()
    gl_rot_dict, gl_pos = recover_root_ang_pos(data.permute(0, 2, 3, 1))
    gl_rot = gl_rot_dict['rot_ang']
    output[:, :1] = gl_rot.permute(0, 3, 1, 2)
    output[:, 1:4] = gl_pos.permute(0, 3, 1, 2)[:, [0, 2, 1]]
    return output


def undo_recover_root_rot_pos(data):
    from data_loaders.humanml.common.quaternion import qrot
    gl_pos = data[..., 1:4][...,[0, 2, 1]]
    gl_rot = data[..., :1]
    rel_pos = torch.zeros_like(gl_pos).to(data.device)
    rel_pos[:, :, 1:, [0, 2]] = gl_pos[:, :, 1:, [0, 2]] - gl_pos[:, :, :-1, [0, 2]]
    gl_quat_rot = torch.zeros(gl_rot.shape[:-1] + (4,)).to(data.device)
    gl_quat_rot[..., :1] = torch.cos(gl_rot)
    gl_quat_rot[..., 2:3] = torch.sin(gl_rot)
    rel_pos = qrot(gl_quat_rot, rel_pos) # rel_pos[:,:,0] is 0 now
    rel_pos[:,:,:-1] = rel_pos[:,:,1:].clone() # very last element of relative positions is lost and the first is not necessarily 0 # try setting it to 0 too
    #rel_pos[:,:,-1] = torch.zeros_like(rel_pos[:,:,-1])
    rel_pos[..., 1] = data[..., 3]
    rel_rot = torch.zeros_like(gl_rot).to(data.device)
    rel_rot[:, :, :-1, :] = gl_rot[:, :, 1:, :] - gl_rot[:, :, :-1, :]
    return rel_rot, rel_pos


def global_to_relative(data):
    """ Convert global root rotation and orientation of motion data to relative.

    Args:
        data (torch.Tensor): Motion data in shape B x n_joints x n_features x n_frames

    Returns:
        torch.Tensor: Motion data where root data is transformed to relative - same shape as input
    """
    output = data.clone()
    rel_rot, rel_pos = undo_recover_root_rot_pos(output.permute(0, 2, 3, 1))
    output[:, :1] = rel_rot.permute(0, 3, 1, 2)
    output[:, 1:4] = rel_pos.permute(0, 3, 1, 2)[:, [0, 2, 1]]
    return output


def grad_fn(x, t, model, model_kwargs):
    """ Computes gradient of model with respect to x

    Args:
        x (torch.Tensor): Noisy data x_t
        t (torch.Tensor): Diffusion step t
        model (MDM): Diffusion model that takes x_t and t as input and outputs hat_x_0

    Returns:
        torch.Tensor: Gradient of hat_x_0(x_t) w.r.t x_t
    """
    inpainting_mask, inpainted_motion = model_kwargs['y']['inpainting_mask'], model_kwargs['y']['inpainted_motion']
    with torch.enable_grad():
        z = x.detach().requires_grad_(True)# * ~inpainting_mask
        hat_x = model(z, t, **model_kwargs)
        assert hat_x.shape == inpainting_mask.shape == inpainted_motion.shape
        # if model_kwargs['y']['global_inpainting']:
        #     hat_x = relative_to_global(hat_x)
        #     inpainted_motion = relative_to_global(inpainted_motion)
        loss = ((inpainted_motion - hat_x).square() * inpainting_mask).sum()
        return torch.autograd.grad(loss, z)[0] * (~inpainting_mask).float(), hat_x


def get_gradient_schedule(schedule_name=None, num_diffusion_steps=1000, scale=.05):
    """ Get gradient schedule for reconstruction guidance
    """
    if schedule_name is None:
        return np.ones(num_diffusion_steps)
    if schedule_name == 'first-half':
        # only add reconstruction guidance to the first half of the diffusion process
        return np.concatenate((np.ones(num_diffusion_steps//2), np.zeros(num_diffusion_steps - num_diffusion_steps//2)))
    if schedule_name == 'last-half':
        # only add reconstruction guidance to the last half of the diffusion process
        return np.concatenate((np.zeros(num_diffusion_steps//2), np.ones(num_diffusion_steps//2)))
    if schedule_name == 'exponential':
        ts = np.arange(num_diffusion_steps)[::-1]
        return np.exp(-scale * ts)
    elif schedule_name == 'sigmoid':
        ts = np.arange(num_diffusion_steps)
        scale /= 5
        return 1 / (1 + np.exp(scale*(-ts + num_diffusion_steps / 2)))
    elif schedule_name == 'half-sigmoid':
        ts = np.arange(num_diffusion_steps)
        scale /= 5
        return 1 / (1 + np.exp(scale*(-ts)))
    else:
        raise NotImplementedError(f"unknown guidance schedule for reconstruction guidance: {schedule_name}")


def requires_reconstruction_guidance(model_kwargs, denoising_step):
    if 'reconstruction_guidance' not in model_kwargs['y'].keys():
        return False
    if model_kwargs['y']['reconstruction_guidance']:
        assert 'stop_recguidance_at' in model_kwargs['y'].keys()
        assert 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys()
        return (denoising_step >= model_kwargs['y']['stop_recguidance_at']).all()
    else:
        return False


def requires_imputation(model_kwargs, denoising_step):
    # FIXME: requires model_kwargs['y']['cutoff'] to be set for inpainting (edit.py, inbetween.py, collect_conditional.py --mask_loss or --recg) and not for pure sampling
    # FIXME: requires model_kwargs['y']['inpainting_mask'] and model_kwargs['y']['inpainted_motion'] to be set for inpainting (edit.py, inbetween.py, collect_conditional.py --mask_loss or --recg) and not for pure sampling
    if 'imputate' not in model_kwargs['y'].keys():
        return False
    if model_kwargs['y']['imputate']:
        assert 'stop_imputation_at' in model_kwargs['y'].keys()
        assert 'inpainting_mask' in model_kwargs['y'].keys() and 'inpainted_motion' in model_kwargs['y'].keys()
        return (denoising_step >= model_kwargs['y']['stop_imputation_at']).all()
    else:
        return False


def load_fixed_dataset(num_samples, data_path=None, multimodal=False, ablation=False):
    data_path = 'save/fixed_dataset/humanml_abs3d'

    input_motions = np.load(os.path.join(data_path, 'input_motions.pt'))
    model_kwargs = np.load(os.path.join(data_path, 'model_kwargs.pt'), allow_pickle=True).item()

    if ablation:
        assert num_samples == 4
        keep_indices = [203, 83, 86, 211]

    else:
        if multimodal and num_samples == 32:
            keep_indices = [2, 14, 18, 40, 45, 49, 65, 66, 73, 78, 83, 86, 87, 99, 105, 114, 115, 116, 119, 142, 145, 156, 158, 160, 167, 176, 192, 197, 207, 211, 213, 255]
        elif not multimodal and num_samples == 5:
            keep_indices = [2, 56, 99, 73, 203]
        elif not multimodal and num_samples == 10:
            keep_indices = [2, 49, 66, 115, 156, 73, 83, 45, 203, 211]
        elif not multimodal and num_samples == 5:
            keep_indices = [45, 78, 115, 142, 176]
        elif not multimodal and num_samples == 1:
            keep_indices = [66]
        elif not multimodal and num_samples == 3:
            keep_indices = [66, 115, 142]
        else:
            raise NotImplementedError(f"Unknown multimodal and num_samples combination for fixed_dataset: {multimodal} and {num_samples}")

    input_motions = torch.Tensor(input_motions[keep_indices])
    model_kwargs['y']['mask'] = model_kwargs['y']['mask'][keep_indices]
    model_kwargs['y']['lengths'] = model_kwargs['y']['lengths'][keep_indices]
    model_kwargs['y']['text'] = [model_kwargs['y']['text'][idx] for idx in keep_indices]
    model_kwargs['y']['tokens'] = [model_kwargs['y']['tokens'][idx] for idx in keep_indices]

    return input_motions, model_kwargs
