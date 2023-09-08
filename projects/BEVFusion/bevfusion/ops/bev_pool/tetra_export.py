import torch
import tetra_hub as hub

import torch.nn.functional as F

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

        # save kept for backward
        # ctx.save_for_backward(kept)

        # # no gradient for geom_feats
        # ctx.mark_non_differentiable(geom_feats)

        return out

    def forward(self, feats, coords, B=1, D=1, H=360, W=360):
        coords = coords * 12
        coords = coords.type(torch.int64)
        coords = torch.clamp(coords, min=0, max=H-1)
        feats = torch.clamp(feats, min=0.5, max=0.5)
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

root_path = "./"
import os

data = torch.load(os.path.join(root_path, "data_input_cpu.pt"))
                      
f = data['feats'].detach().cpu()
c = data['coords'].detach().cpu()
traced_model = torch.jit.trace(m, (f, c))


hub.submit_profile_job(traced_model, name="bev_pool",
                    device=hub.Device(name="Samsung Galaxy S23 Ultra"),
                    # device=hub.Device(name="Apple iPhone 14 Pro"),
                    input_shapes={"feats" : (1835478, 80), "coords": (1835478, 4)})