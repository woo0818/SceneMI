import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import clip
from model.rotation2xyz import Rotation2xyz

from model.net_modules import MLPNet
from typing import Optional, Tuple
from einops.layers.torch import Rearrange
import einops
import math
from vit_pytorch import ViT
import smplx


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self,
                 inp_channels,
                 out_channels,
                 kernel_size,
                 n_groups=8,
                 zero=False):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            # adding the height dimension for group norm
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

        if zero:
            # zero init the convolution
            nn.init.zeros_(self.block[0].weight)
            nn.init.zeros_(self.block[0].bias)

    def forward(self, x):
        """
        Args:
            x: [n, c, l]
        """
        return self.block(x)


class Conv1dAdaGNBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(inp_channels,
                      out_channels,
                      kernel_size,
                      padding=kernel_size // 2),
            # adding the height dimension for group norm
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
        )
        self.block2 = nn.Mish()

    def forward(self, x, c):
        """
        Args:
            x: [n, nfeat, l]
            c: [n, ncond, 1]
        """
        scale, shift = c.chunk(2, dim=1)
        x = self.block1(x)
        x = ada_shift_scale(x, shift, scale)
        x = self.block2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kargs):
        return self.fn(x, *args, **kargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads
                                       ), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)


def ada_shift_scale(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualTemporalBlock(nn.Module):
    def __init__(self,
                 inp_channels,
                 out_channels,
                 embed_dim,
                 kernel_size=5,
                 adagn=False,
                 zero=False):
        super().__init__()
        self.adagn = adagn

        self.blocks = nn.ModuleList([
            # adagn only the first conv (following guided-diffusion)
            (Conv1dAdaGNBlock(inp_channels, out_channels, kernel_size) if adagn
             else Conv1dBlock(inp_channels, out_channels, kernel_size)),
            Conv1dBlock(out_channels, out_channels, kernel_size, zero=zero),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            # adagn = scale and shift
            nn.Linear(embed_dim, out_channels * 2 if adagn else out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        if adagn:
            # zero the linear layer in the time_mlp so that the default behaviour is identity
            nn.init.zeros_(self.time_mlp[1].weight)
            nn.init.zeros_(self.time_mlp[1].bias)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        cond = self.time_mlp(t)
        if self.adagn:
            # using adagn
            out = self.blocks[0](x, cond)
        else:
            # using addition
            out = self.blocks[0](x) + cond
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
            self,
            input_dim,
            cond_dim,
            dim=256,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            adagn=False,
            zero=False,
            added_input_channels=0,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print('dims: ', dims, 'mults: ', dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')
        print(f'[ models/temporal ] added_input_channels: {added_input_channels}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            is_first = ind == 0

            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_in+added_input_channels*is_first,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_out,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                    if attention else nn.Identity(),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)
        self.mid_attn = Residual(
            PreNorm(mid_dim,
                    LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_out * 2,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_in,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                    if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        # use the last dim_in to support the case where the mult doesn't start with 1.
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5),
            nn.Conv1d(dim_in, input_dim, 1),
        )

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

    def forward(self, x, cond):
        '''
            x : [ seqlen x batch x dim ]
            cons: [ batch x cond_dim]
        '''

        x = einops.rearrange(x, 's b d -> b d s')
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        # print('x:', x.shape)

        x = einops.rearrange(x, 'b d s -> s b d')
        return x


def cal_concat_multiple(in1, in2, multiple):
    """
    calculate the output channels of the concatenation of the two inputs while keeping the output channels a multiple of the given number
    """
    a = (in1 + in2) / multiple
    return int((1 - (a - math.floor(a))) * multiple + in1 + in2)


class TemporalUnetLarge(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=256,
        dim_mults=(1, 2, 4, 8),
        out_mult=8,
        attention=False,
        adagn=False,
        zero=False,
        added_input_channels=0,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        print('dims: ', dims, 'mults: ', dim_mults)
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            # SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            is_first = ind == 0

            self.downs.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_in+added_input_channels*is_first,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_out,
                                          dim_out,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                    if attention else nn.Identity(),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)
        self.mid_attn = Residual(
            PreNorm(mid_dim,
                    LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim,
                                                mid_dim,
                                                embed_dim=time_dim,
                                                adagn=adagn,
                                                zero=zero)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # print(dim_out, dim_in)
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList([
                    ResidualTemporalBlock(dim_out * 2,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    ResidualTemporalBlock(dim_in,
                                          dim_in,
                                          embed_dim=time_dim,
                                          adagn=adagn,
                                          zero=zero),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                    if attention else nn.Identity(),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))

        # use the last dim_in to support the case where the mult doesn't start with 1.
        final_in = cal_concat_multiple(dim_in, input_dim, out_mult)
        # only temporary
        final_type = 4  # NOTE: flag.LARGE_OUT_TYPE
        if final_type == 1:
            print('using final type 1')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim, input_dim, 1,
                          groups=input_dim),
            )
        elif final_type == 2:
            # more kernels of size 5
            print('using final type 2')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 1),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim,
                          input_dim,
                          5,
                          padding=2,
                          groups=input_dim),
            )
        elif final_type == 3:
            # all kernels of size 5
            print('using final type 3')
            self.final_conv = nn.Sequential(
                # combine the skip connection with the upstream feature
                # randomly arrange the channels
                nn.Conv1d(dim_in + input_dim, final_in, 5, padding=2),
                # [batch, mult * in_dim, seqlen]
                nn.Conv1d(final_in,
                          out_mult * input_dim,
                          5,
                          padding=2,
                          groups=out_mult),
                nn.Mish(),
                nn.Conv1d(out_mult * input_dim,
                          input_dim,
                          5,
                          padding=2,
                          groups=input_dim),
            )
        else:
            raise NotImplementedError()

        if zero:
            # zero the convolution in the final conv
            nn.init.zeros_(self.final_conv[-1].weight)
            nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, x, cond):
        '''
            x : [ seqlen x batch x dim ]
            cons: [ batch x cond_dim]
        '''

        x = einops.rearrange(x, 's b d -> b d s')
        src = x
        # print('x:', x.shape)

        c = self.time_mlp(cond)
        # print('c:', c.shape)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, c)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, c)
            x = resnet2(x, c)
            x = attn(x)
            x = upsample(x)

        # [batch, last_dim + in_dim, seqlen]
        x = torch.concat([x, src], dim=1)
        # [batch, in_dim, seqlen]
        x = self.final_conv(x)
        # print('x:', x.shape)

        x = einops.rearrange(x, 'b d s -> s b d')
        return x


class MDM_Scene_UNET(nn.Module):
    """
    Diffuser's style UNET
    """
    def __init__(self,
                 modeltype,
                 njoints,
                 nfeats,
                 num_actions,
                 translation,
                 pose_rep,
                 glob,
                 glob_rot,
                 latent_dim=512,
                 dim_mults=(1, 2, 4, 8),
                 attention=False,
                 ablation=None,
                 legacy=False,
                 data_rep='rot6d',
                 dataset='amass',
                 clip_dim=512,
                 emb_trans_dec=False,
                 clip_version=None,
                 adagn=False,
                 zero=False,
                 arch='unet',
                 unet_out_mult=8,
                 xz_only=False,
                 train_keypoint_mask='none',
                 keyframe_conditioned=False,
                 keyframe_selection_scheme='in-between',
                 zero_keyframe_loss=False,
                 imputation = 'all',
                 scene_type = 'height_map',
                 light_bps = True,
                 sub_bps = 0,
                 free_p = 0.1,
                 nframes = 121,
                 scene_size = 48,
                 imputation_timestep=30,
                 **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.light_bps = light_bps
        self.sub_bps = sub_bps

        self.imputation = imputation
        self.imputation_timestep = imputation_timestep
        self.time_weight = self.mask_weight(self.imputation)

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        print('MDM latent dim: ', latent_dim)
        self.latent_dim = latent_dim
        self.dim_mults = dim_mults
        self.attention = attention

        self.ablation = ablation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats
        if self.light_bps:
            self.cond_feats = 67 * 3
        elif self.sub_bps > 0:
            self.cond_feats = self.sub_bps * 3
        
        #self.masked_attention = False
        #if self.masked_attention:
        self.single_mask = torch.ones((1, 1), dtype=torch.bool).to('cuda')

        self.train_keypoint_mask = train_keypoint_mask

        self.wo_frame_feature = kargs['wo_frame_feature']
        self.wo_scene_feature = kargs['wo_scene_feature']

        self.cond_beta = kargs['beta']
        self.body_abstract= kargs['body_abstract'] 

        self.scene_type = scene_type
        self.free_p = free_p

        self.better_cond = True
        self.latent_dim_input = self.latent_dim
        
        if self.wo_frame_feature:
            self.cond_latent_dim = 0
        else:
            self.cond_latent_dim = 32
            if self.light_bps:
                self.cond_process = MLPNet(activation='mish', input_size = self.cond_feats, 
                              hid_layer = None, 
                              output_size = self.cond_latent_dim)
            
            else:
                self.cond_process = MLPNet(activation='mish', input_size = self.cond_feats, 
                              hid_layer = [1024, 256, 32], 
                              output_size = self.cond_latent_dim)

        if self.cond_beta:
            assert self.body_abstract == "part_all"
            shape_input_size = 7
            self.beta_process = MLPNet(activation='mish', input_size = shape_input_size, 
                              hid_layer = None, 
                              output_size = self.latent_dim)
            
        self.keyframe_conditioned = keyframe_conditioned
        self.keyframe_selection_schedme = keyframe_selection_scheme
        self.zero_keyframe_loss = zero_keyframe_loss

        self.train_keypoint_mask = train_keypoint_mask
        

        self.add_mask = True
        if self.add_mask:
            added_channels = self.cond_latent_dim + 1
        else:
            added_channels = self.cond_latent_dim


        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)

        # no need for the input process any
        # NOTE: just a mock, doing nothing
        # self.input_process = InputProcess(self.data_rep, self.input_feats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim,
                                                       dropout=0)
        self.emb_trans_dec = emb_trans_dec

        print('Using UNET with lantent dim: ', self.latent_dim, ' and mults: ',
              self.dim_mults)
        if arch == 'unet':
            self.unet = TemporalUnet(input_dim=self.input_feats,
                                     cond_dim=self.latent_dim,
                                     dim=self.latent_dim,
                                     dim_mults=self.dim_mults,
                                     attention=self.attention,
                                     adagn=adagn,
                                     zero=zero,
                                     added_input_channels=added_channels,)
        elif arch == 'unet_large':
            print('UNET large variation with output multiplier: ',
                  unet_out_mult)
            self.unet = TemporalUnetLarge(input_dim=self.input_feats,
                                          cond_dim=self.latent_dim,
                                          dim=self.latent_dim,
                                          dim_mults=self.dim_mults,
                                          attention=self.attention,
                                          adagn=adagn,
                                          zero=zero,
                                          out_mult=unet_out_mult,
                                          added_input_channels=added_channels,)
        else:
            raise NotImplementedError()

        self.embed_timestep = TimestepEmbedder(self.latent_dim,
                                               self.sequence_pos_encoder)

        assert self.scene_type == "occ_map24"
        self.scene_channels = 24
        self.scene_size = scene_size

        if not self.wo_scene_feature:
            self.scene_embedding = ViT(
                    image_size=self.scene_size,
                    patch_size=self.scene_size // 4,
                    channels=self.scene_channels,
                    num_classes=self.latent_dim,
                    dim=1024,
                    depth=6,
                    heads=16,
                    mlp_dim=2048,
                    dropout=0.1,
                    emb_dropout=0.1
                )

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()
            if not name.startswith('clip_model.')
        ]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(
            clip_version, device='cpu',
            jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model
        )  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in [
            'humanml', 'kit'
        ] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2  # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(
                raw_text, context_length=context_length, truncate=True
            ).to(
                device
            )  # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(
                device
            )  # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def mask_weight(self, imputation):
        t1 = self.imputation_timestep
        step = 1000
        time_weight = torch.ones((step,1,1,1)).float().cuda()

        if imputation == "all":
            return time_weight

        elif imputation == "early":
            time_weight[:t1] = 0.0
            return time_weight


    def forward(self, x, timesteps, y=None, obs_x0=None, obs_mask=None, bps_sbj_mask=None):
        """
        Args:
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            y: dict of conditioning information
            obs_mask: [batch_size, njoints, nfeats, max_frames], mask for the observed keyframes

        Returns: [batch_size, njoints, nfeats, max_frames]
        """
        # imputation happened in here!

        obs_mask_apply = obs_mask.clone()
        obs_mask_apply[:,:,:,1:-1] = obs_mask_apply[:,:,:,1:-1] * self.time_weight[timesteps]
        x = obs_x0 * obs_mask_apply.float() + x * (1 - obs_mask_apply.float())

        x = torch.cat([x, obs_mask[:,[0],:,:]], dim=1)

        cond = y['occ_map']

        y['bps_sbj'] = y['bps_sbj'] * bps_sbj_mask

        return self.forward_core(x, cond, timesteps, y, obs_mask)

    def forward_core(self, x, cond, timesteps, y=None, obs_mask=None):
        """
        Args:
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)

        Returns: [batch_size, njoints, nfeats, max_frames]
        """
        bs, njoints, nfeats, nframes = x.shape
        t_emb = self.embed_timestep(timesteps)   # [1, b, d]

        sampling = y.get('sampling', False)
        if not self.wo_scene_feature:
            scene_emb = self.scene_embedding(cond).unsqueeze(0)  # [1, b, d]

            if sampling:
                force_mask = y.get('uncond', False)
                if force_mask:
                    free_ind = torch.ones(scene_emb.shape[1]).bool().to(scene_emb.device)
                else:
                    free_ind = torch.zeros(scene_emb.shape[1]).bool().to(scene_emb.device)
            else:
                free_ind = torch.rand(scene_emb.shape[1]).to(scene_emb.device) < self.free_p
            scene_emb[:,free_ind] = 0.

            emb = t_emb + scene_emb
            
        else:
            emb = t_emb

        emb = emb.squeeze(0)  # [bs, d]
        
        if self.cond_beta:
            beta_emb = self.beta_process(y['body_abstract'])    
            beta_emb = beta_emb.squeeze(1)
            emb = emb + beta_emb
       
        # no need for positional embedding in the input becuase we use convolution.
        # [nframes, bs, d]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        
        if not self.wo_frame_feature:
            frame_feature = y['bps_sbj'].permute(0,3,1,2).reshape(bs, nframes, -1)

            frame_emb = self.cond_process(frame_feature).permute(1, 0, 2) #.permute(0,2,1).unsqueeze(2) [bs, d=128, nfeats, nframes]
            x = torch.cat((x, frame_emb), axis=2)   # [seqlen, bs, input_d + cond_d]
        
        assert nframes == 121 #, f"the input should be 121 frames not {nframes}"
        x = F.pad(x, (0, 0, 0, 0, 0, 128 - nframes), value=0)

        x = self.unet(
            x,
            cond=emb)  # , src_key_padding_mask=~maskseq)  # [nframes, bs, d]
        # remove the padding from the output
        # [nframes, bs, d]

        x = x[:nframes]
        x = x.reshape(nframes, bs, self.njoints, nfeats)

        # [bs, njoints, nfeats, nframes]
        x = x.permute(1, 2, 3, 0).float()
        return x

    def _apply(self, fn):
        super()._apply(fn)
        #self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        #self.rot2xyz.smpl_model.train(*args, **kargs)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        # output is the input
        # self.latent_dim = latent_dim
        # self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        # if self.data_rep == 'rot_vel':
        #     self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            # just use what's in the input
            # x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            raise NotImplementedError()
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats, xz_only):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = 2 if xz_only else njoints
        self.nfeats = nfeats
        # NOTE: just pass through it
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            # output = self.poseFinal(output)  # [seqlen, bs, 150]
            pass
        elif self.data_rep == 'rot_vel':
            pass
            # first_pose = output[[0]]  # [1, bs, d]
            # first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            # vel = output[1:]  # [seqlen-1, bs, d]
            # vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            # output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output