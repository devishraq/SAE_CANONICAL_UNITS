import torch
import torch.nn.functional as F

def stitching_novel_fraction(sae_small, sae_large, acts, thresh=0.7, n_bootstrap=200):
    device = acts.device
    with torch.no_grad():
        z_s = sae_small.encode(acts)
        x_s = sae_small.decode(z_s)
        E = acts - x_s
        base_mse = (E**2).mean().item()

        small_n = F.normalize(sae_small.W_dec, dim=1)
        large_n = F.normalize(sae_large.W_dec, dim=1)
        sim = large_n @ small_n.T
        max_sim, _ = sim.max(dim=1)
        candidate_mask = max_sim < thresh
        cand_idx = torch.where(candidate_mask)[0]
        
        if len(cand_idx) == 0:
            return {"novel_fraction": 0.0, "base_mse": base_mse, "n_candidates": 0, "ci_low": 0, "ci_high": 0}

        z_l = sae_large.encode(acts)
        z_cand = z_l[:, cand_idx]
        dec_cand = sae_large.W_dec[cand_idx]

        z_norm2 = (z_cand**2).sum(0)
        dec_norm2 = (dec_cand**2).sum(1)
        norm_C2 = z_norm2 * dec_norm2

        S = torch.einsum("nc,nd->cd", z_cand, E)
        dot_EC = (S * dec_cand).sum(1)

        improves = (2 * dot_EC) > norm_C2
        n_novel = improves.sum().item()
        novel_frac = n_novel / sae_large.W_dec.shape[0]

        N = acts.shape[0]
        boot = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, N, (N,), device=device)
            Eb = E[idx]
            z_cb = z_cand[idx]
            Sb = torch.einsum("nc,nd->cd", z_cb, Eb)
            dot_b = (Sb * dec_cand).sum(1)
            norm_b = (z_cb**2).sum(0) * dec_norm2
            boot.append(((2 * dot_b) > norm_b).sum().item() / sae_large.W_dec.shape[0])
        
        boot = torch.tensor(boot)
        ci_low, ci_high = torch.quantile(boot, torch.tensor([0.025, 0.975])).tolist()

    return {
        "novel_fraction": novel_frac,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_candidates_geo": len(cand_idx),
        "n_novel_func": n_novel,
        "base_mse": base_mse,
        "max_sims": max_sim.cpu().numpy()
    }