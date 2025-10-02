import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'scene' in notnone_batches[0]:
        scenebatch = [b['scene'] for b in notnone_batches]
        cond['y'].update({'scene': collate_tensors(scenebatch)})

    if 'gt_joints' in notnone_batches[0]:
        scenebatch = [b['gt_joints'] for b in notnone_batches]
        cond['y'].update({'gt_joints': collate_tensors(scenebatch)})

    if 'bps_sbj' in notnone_batches[0]:
        bps_sbj_batch = [b['bps_sbj'] for b in notnone_batches]
        cond['y'].update({'bps_sbj': collate_tensors(bps_sbj_batch)})

    if 'occ_map' in notnone_batches[0]:
        occ_map_batch = [b['occ_map'] for b in notnone_batches]
        cond['y'].update({'occ_map': collate_tensors(occ_map_batch)})
    
    if 'gt' in notnone_batches[0]:
        gt_databatch = [b['gt'] for b in notnone_batches]
        cond['y'].update({'gt': collate_tensors(gt_databatch)})

    if 'scene_info' in notnone_batches[0]:
        scene_info = [b['scene_info']for b in notnone_batches]
        cond['y'].update({'scene_info': scene_info})

    if 'betas' in notnone_batches[0]:
        betas_info = [b['betas']for b in notnone_batches]
        cond['y'].update({'betas': collate_tensors(betas_info)})

    if 'body_abstract' in notnone_batches[0]:
        body_abstract_info = [b['body_abstract']for b in notnone_batches]
        cond['y'].update({'body_abstract': collate_tensors(body_abstract_info)})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

def amass_collate(batch):
    adapted_batch = [{
        'inp': (b.clone().detach().T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
    } for b in batch]
    return collate(adapted_batch)

def trumans_collate(batch):
    adapted_batch = [{
        'inp': (b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'gt': (b[1].T).float().unsqueeze(1),
        'gt_joints': b[2],
        'bps_sbj': b[3].float().permute(1,2,0),
        'occ_map': b[4].float(),
        'scene_info': b[5],
        'betas': b[6].float(),
        'body_abstract': b[7].float()
    } for b in batch]
    return collate(adapted_batch)
