# by pcli2
import torch
import torch.jit as jit
from typing import List, Tuple, Dict
import sys
import scipy.io as sio

#@jit.ignore
def clip_mask(mask: torch.Tensor, clip_length: int, dim: int):
    #因为在网络的每一层中,会有不同帧率的表示, clip_length为当前的帧数的表示,相对于原始的mask的长度可能是N分之一
    assert mask.shape[dim] % clip_length == 0, "mask dim {0} must be a integer multiple of clip_length {1}".format(mask.shape[dim], clip_length)#帧率必须为整数
    nmod = int(mask.shape[dim] / clip_length)#计算帧率的倍数
    indices = torch.Tensor([i*nmod for i in range(clip_length)]).long()
    indices = indices.to(mask.device)
    #跳帧选取当前帧率 每一帧组合的mask信息
    cliped_mask = torch.index_select(mask, dim, indices).contiguous()
    return cliped_mask


@jit.ignore
def cnn2rnn(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 4 and x.shape[2] == 1, "x must be 4-dim tensor and dim 2 must be 1"
    x = x.squeeze(2)
    x = x.permute(2, 0, 1)
    x = x.contiguous()
    return x


@jit.ignore
def rnn2cnn(x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) == 3, "x must be 3-dim tensor"
    x = x.permute(1, 2, 0)
    x = x.unsqueeze(2)
    x = x.contiguous()
    return x

@jit.ignore
def clip_chunk(x: torch.Tensor, win_cur: int, win_fur: int, meta:Dict[str, torch.Tensor]):
    n, c, h, w = x.size()
    x = x.reshape(n, c, -1, win_cur)
    x_fur = x[:, :, 1:, :win_fur].clone()
    x_fur = torch.cat((x_fur, x_fur.new_zeros((n, c, 1, win_fur)).cuda()), dim=2)
    x = torch.cat((x, x_fur), dim=3).reshape(n, c, 1, -1)

    mask_chunk = clip_mask(meta["mask"].clone(), w, 1) # (b, t)
    mask_chunk = mask_chunk.unsqueeze(1).reshape(n, -1, win_cur) # (b, d, t)
    mask_fur = mask_chunk[:, 1:, :win_fur].clone()
    mask_fur = torch.cat((-1*mask_fur, mask_fur.new_zeros((n, 1, win_fur)).cuda()), dim=1)
    mask_chunk =  torch.cat((mask_chunk, mask_fur), dim=2).reshape(n, 1, -1).squeeze(1)  # (b, t)
    meta["mask_chunk"] = mask_chunk
    return x