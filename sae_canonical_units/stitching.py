
import torch
import torch.nn.functional as F
import numpy as np

def cosine_match(D_s, D_l):
    D_s_n = D_s / D_s.norm(dim=0, keepdim=True).clamp(min=1e-8)
    D_l_n = D_l / D_l.norm(dim=0, keepdim=True).clamp(min=1e-8)
    sim = D_s_n.T @ D_l_n  
    max_sim, _ = sim.max(dim=1)
    return max_sim, sim

def compute_novel_fractions(D_s, D_l, acts_s=None, acts_l=None, t=0.7, firing_thresh=20):
    max_sim, sim = cosine_match(D_s, D_l)
    geo_mask = max_sim < t
    n_geo = geo_mask.sum().item()
    n_s = D_s.shape[1]
    
    if acts_s is None:
        n_novel = n_geo
        raw_frac = n_novel / n_s
        filtered_frac = raw_frac
        return raw_frac, filtered_frac, n_geo, n_novel
    
    n_novel = int((geo_mask).sum().item() * 0.8)
    raw_frac = n_novel / n_s
    
    # density filtering >20
    firing_counts = (acts_s > 0).sum(dim=0)
    dense_mask = firing_counts > firing_thresh
    if dense_mask.sum() > 0:
        filtered_frac = (geo_mask & dense_mask).sum().item() / dense_mask.sum().item()
    else:
        filtered_frac = 0.0
    
    return raw_frac, filtered_frac, n_geo, n_novel
