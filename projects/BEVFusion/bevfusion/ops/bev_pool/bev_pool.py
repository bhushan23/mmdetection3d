import torch

from . import bev_pool_ext

from math import log10, sqrt
import numpy as np
import torch
import torch.nn.functional as F
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

class QuickCumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        """
            1. call inclusive cumsum
            2. compute lenght and compute prefix sum for each interval
            3. flatten out feature indices for destination data
            4. build indices from interval start and end for source data
            5. map data from source data (5) to destination data (3)
        """
        x_prefix = x.cumsum(0)
        kept = torch.ones(x_prefix.shape[0], device=x_prefix.device, dtype=torch.bool)

        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].type(torch.int64)
        interval_lengths = interval_starts[1:] - interval_starts[:-1]
        
        interval_ends = interval_starts[:-1] + interval_lengths - 1
        interval_ends = F.pad(interval_ends, (0, 1), value=x_prefix.shape[0]-1)

        x_prefix = x_prefix[interval_ends] - x_prefix[interval_starts] + x[interval_starts]

        gf_W, gf_H, gf_D, gf_B = torch.split(geom_feats, split_size_or_sections=[1, 1, 1, 1], dim=1)
        geom_feats = gf_W.squeeze(1) + gf_H.squeeze(1) * W + gf_D.squeeze(1) * H + gf_B.squeeze(1)
        geom_feats = geom_feats.type(torch.int64)
        geom_feats = geom_feats[interval_starts]

        C = x_prefix.shape[-1]
        out = torch.zeros((B*D*H*W, C), device=x_prefix.device, dtype=x_prefix.dtype)
        out[geom_feats, :] += x_prefix
        out = out.reshape((B, D, H, W, C))
        return out

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept, ) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        geom_feats = geom_feats.int()

        out = bev_pool_ext.bev_pool_forward(
            x,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes

        out_grad = out_grad.contiguous()
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad,
            geom_feats,
            interval_lengths,
            interval_starts,
            B,
            D,
            H,
            W,
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    # NOTE: without following clamp,
    # TFLite model fails for sub-sequent loads.
    # This is probably due to random data being fed by profiler.
    # As part of model pipeline, this might work. But will have to check.
    # coords = coords * 12
    # coords = coords.type(torch.int64)
    # coords = torch.clamp(coords, min=0, max=H-1)
    # feats = torch.clamp(feats, min=0.5, max=0.5)

    ranks = (
        coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
        coords[:, 2] * B + coords[:, 3])
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsum.apply(feats, coords, ranks, B, D, H, W)

    # Quick check for PSNR
    # x_cuda = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    # x_cuda_np = x_cuda.detach().cpu().numpy()
    # x_ours_np = x.detach().cpu().numpy()
    # print(PSNR(x_cuda_np, x_ours_np))

    # NOTE: Work-around to slice input in expected output shape
    # out_size = B * D * H * W
    # x_sliced = feats[:out_size, :]
    # x = x_sliced.reshape((B, D, H, W, feats.shape[-1]))
    
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x

