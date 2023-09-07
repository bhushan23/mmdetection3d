import torch
import tetra_hub as hub
# from .bev_pool import bev_pool

# Export 
class QCS(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def cum_sum(self, x, geom_feats, ranks, B, D, H, W):
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

    def forward(self, feats, coords, B=1, D=1, H=360, W=360):
        ranks = (
            coords[:, 0] * (W * D * B) + coords[:, 1] * (D * B) +
            coords[:, 2] * B + coords[:, 3])
        indices = ranks.argsort()
        feats, coords, ranks = feats[indices], coords[indices], ranks[indices]
        return self.cum_sum(feats, coords, ranks, B, D, H, W)

m = QCS()

m.eval()
m.cpu()
m = m.to('cpu')

root_path = "/home/bhushan/work/bev-fusion/repo/mmdetection3d/"
import os

data = torch.load(os.path.join(root_path, "bev_pool_data.pt"))
                      
f = data['feats'].detach().cpu()
c = data['coords'].detach().cpu()
traced_model = torch.jit.trace(m, (f, c))

print(traced_model.graph)
print(traced_model.code)
# exit(0)

hub.submit_profile_job(traced_model, name="bev_pool",
                    device=hub.Device(name="Samsung Galaxy S23 Ultra"),
                    # device=hub.Device(name="Apple iPhone 14 Pro"),
                    input_shapes={"feats" : (1835478, 80), "coords": (1835478, 4)})