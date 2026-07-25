import copy
import torch
import torch.nn.functional as F
from.stitching import stitching_novel_fraction

def random_decoder_control(sae_small, sae_large, acts, thresh=0.7):
    print("Running Isotropic Random Control...")
    device = sae_large.W_dec.device
    acts = acts.to(device).float()

    W = sae_large.W_dec.float().clone()
    norms = W.norm(dim=1, keepdim=True)
    rand = torch.randn_like(W)
    rand = rand / rand.norm(dim=1, keepdim=True) * norms

    small_n = F.normalize(sae_small.W_dec.float(), dim=1)
    rand_n = F.normalize(rand, dim=1)
    sim = rand_n @ small_n.T
    max_sim = sim.max(dim=1).values
    geo_frac = (max_sim < thresh).float().mean().item()

    with torch.no_grad():
        z_s = sae_small.encode(acts).float()
        x_s = sae_small.decode(z_s).float()
        E = acts - x_s
        z_l = sae_large.encode(acts).float()
        cand_idx = torch.where(max_sim < thresh)[0]
        if len(cand_idx) == 0:
            return {"isotropic_geo_novel_frac": geo_frac, "isotropic_func_novel_frac": 0.0}
        z_cand = z_l[:, cand_idx]
        dec_cand = rand[cand_idx]
        z_norm2 = (z_cand**2).sum(0)
        dec_norm2 = (dec_cand**2).sum(1)
        norm_C2 = z_norm2 * dec_norm2
        dot_EC = (z_cand * (E @ dec_cand.T)).sum(0)
        improves = (2 * dot_EC - norm_C2) > 0.0
        func_frac = improves.sum().item() / sae_large.W_dec.shape[0]
    return {
        "isotropic_geo_novel_frac": geo_frac,
        "isotropic_func_novel_frac": func_frac,
    }

def shuffled_decoder_control(sae_small, sae_large, acts, thresh=0.7):
    print("Running Shuffled Decoder Control...")
    device = sae_large.W_dec.device
    acts = acts.to(device).float()

    # FIX: permute ROWS (n_latents), not columns
    perm = torch.randperm(sae_large.W_dec.shape[0])
    W_shuf = sae_large.W_dec[perm, :].clone()

    small_n = F.normalize(sae_small.W_dec.float(), dim=1)
    shuf_n = F.normalize(W_shuf.float(), dim=1)
    sim = shuf_n @ small_n.T
    max_sim = sim.max(dim=1).values
    geo_frac = (max_sim < thresh).float().mean().item()

    with torch.no_grad():
        z_s = sae_small.encode(acts).float()
        x_s = sae_small.decode(z_s).float()
        E = acts - x_s
        z_l = sae_large.encode(acts).float()
        cand_idx = torch.where(max_sim < thresh)[0]
        if len(cand_idx) == 0:
            return {"shuffled_geo_novel_frac": geo_frac, "shuffled_func_novel_frac": 0.0}
        z_cand = z_l[:, cand_idx]
        dec_cand = W_shuf[cand_idx]
        z_norm2 = (z_cand**2).sum(0)
        dec_norm2 = (dec_cand**2).sum(1)
        norm_C2 = z_norm2 * dec_norm2
        dot_EC = (z_cand * (E @ dec_cand.T)).sum(0)
        improves = (2 * dot_EC - norm_C2) > 0.0
        func_frac = improves.sum().item() / sae_large.W_dec.shape[0]
    return {
        "shuffled_geo_novel_frac": geo_frac,
        "shuffled_func_novel_frac": func_frac,
    }