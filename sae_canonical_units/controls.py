import copy
import torch
import torch.nn.functional as F
from .stitching import stitching_novel_fraction

def random_decoder_control(sae_small, sae_large, acts, thresh=0.7):
    sae_shuffled = copy.deepcopy(sae_large)
    
    perm = torch.randperm(sae_shuffled.W_dec.shape[0], device=sae_shuffled.W_dec.device)
    sae_shuffled.W_dec.data = sae_shuffled.W_dec.data[perm]
    
    small_n = F.normalize(sae_small.W_dec, dim=1)
    shuffled_n = F.normalize(sae_shuffled.W_dec, dim=1)
    sim = shuffled_n @ small_n.T
    max_sim, _ = sim.max(dim=1)
    geo_frac = (max_sim < thresh).float().mean().item()
    
    res = stitching_novel_fraction(sae_small, sae_shuffled, acts, thresh=thresh, n_bootstrap=100)
    func_frac = res["novel_fraction"]
    
    return {
        "shuffled_geo_novel_frac": geo_frac,
        "shuffled_func_novel_frac": func_frac,
        "note": "Geo should be ~99%, Func should be ~0%. Proves metric isn't random."
    }

def same_width_control(sae_large_a, sae_large_b, acts, thresh=0.7):
    if sae_large_b is None:
        return {"control_note": "no second 16k checkpoint found"}
    res = stitching_novel_fraction(sae_large_a, sae_large_b, acts, thresh=thresh, n_bootstrap=200)
    return {
        "same_width_novel_frac": res["novel_fraction"], 
        "same_width_ci": (res["ci_low"], res["ci_high"])
    }