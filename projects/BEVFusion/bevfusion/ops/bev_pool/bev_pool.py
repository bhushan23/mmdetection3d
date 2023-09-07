import torch

from . import bev_pool_ext


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
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)

        kept[1:] = ranks[1:] != ranks[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = x.shape[0] - interval_starts[-1]
        interval_ends = interval_starts + interval_lengths
        interval_ends[-1] -= 1

        interval_ends = interval_ends.type(torch.int64)
        interval_starts = interval_starts.type(torch.int64)
        last_val = x[-1]
        x = x[interval_ends] - x[interval_starts]
        x[-1] = last_val

        gf_W, gf_H, gf_D, gf_B = torch.split(geom_feats, split_size_or_sections=[1, 1, 1, 1], dim=1)
        geom_feats = gf_W.squeeze(1) + gf_H.squeeze(1) * W + gf_D.squeeze(1) * H + gf_B.squeeze(1)
        geom_feats = geom_feats.type(torch.int64)

        C = x.shape[-1]
        out = torch.zeros((B*D*H*W, C), device=x.device, dtype=x.dtype)

        x = torch.repeat_interleave(x, interval_lengths, dim=0)
        out[geom_feats, :] += x
        out = out.reshape((B, D, H, W, C))    

        # save kept for backward
        # ctx.save_for_backward(kept)

        # # no gradient for geom_feats
        # ctx.mark_non_differentiable(geom_feats)

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

    # ranks = (
    #     coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
    #     coords[:, 2] * B + coords[:, 3])
    # indices = ranks.argsort()
    # feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    # # x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    # x = QuickCumsum.apply(feats, coords, ranks, B, D, H, W)
    out_size = B * D * H * W
    x_sliced = feats[:out_size, :]
    x = x_sliced.reshape((B, D, H, W, feats.shape[-1]))
    
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


