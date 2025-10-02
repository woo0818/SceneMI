
import numpy as np
import torch
import logging

import chamfer_distance as chd

import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
):
    """
    signed distance between two pointclouds
    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
    Returns:
        - y2x: Torch.Tensor
            the displacement vector from y to x
        - x2y: Torch.Tensor
            the displacement vector from x to y
    """

    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near
    y2x = y - y_near

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    return -y2x, -x2y