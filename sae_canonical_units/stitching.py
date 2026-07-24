import torch
import torch.nn.functional as F

def stitching_novel_fraction(sae_small, sae_large, acts, thresh=0.7, n_bootstrap=200, min_improvement=0.0):
    device = sae_small.W_dec.device
    acts = acts.to(device)
    with torch.no_grad():
        z_s = sae_small.encode(acts)
        x_s = sae_small.decode(z_s)
        E = acts - x_s

        base_mse = (E**2).mean().item()
        acts_mean = acts.mean(dim=0)
        acts_var = ((acts - acts_mean)**2).mean().item()
        acts_mean_sq = (acts**2).mean().item()
        nmse = base_mse / (acts_mean_sq + 1e-8)
        explained_variance = 1.0 - base_mse / (acts_var + 1e-8)
        rms_x = (acts_mean_sq ** 0.5)
        rmse = (base_mse ** 0.5)

        small_n = F.normalize(sae_small.W_dec, dim=1)
        large_n = F.normalize(sae_large.W_dec, dim=1)
        
        # CHUNKED SIMILARITY COMPUTATION TO PREVENT T4 OOM
        max_sim = torch.empty(large_n.shape[0], device=device)
        chunk_size = 8192
        for i in range(0, large_n.shape[0], chunk_size):
            sim_chunk = large_n[i:i+chunk_size] @ small_n.T
            max_sim[i:i+chunk_size] = sim_chunk.max(dim=1).values
            
        cand_idx = torch.where(max_sim < thresh)[0]

        if len(cand_idx) == 0:
            return {
                "novel_fraction": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_candidates_geo": 0, "n_novel_func": 0,
                "base_mse": base_mse, "nmse": nmse,
                "explained_variance": explained_variance,
                "rms_act": rms_x, "rmse": rmse
            }

        z_l = sae_large.encode(acts)
        z_cand = z_l[:, cand_idx]
        dec_cand = sae_large.W_dec[cand_idx]

        z_norm2 = (z_cand**2).sum(0)
        dec_norm2 = (dec_cand**2).sum(1)
        norm_C2 = z_norm2 * dec_norm2
        S = torch.einsum("nc,nd->cd", z_cand, E)
        dot_EC = (S * dec_cand).sum(1)
        improves = (2*dot_EC - norm_C2) > min_improvement

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
            boot.append(((2*dot_b - norm_b) > min_improvement).sum().item() / sae_large.W_dec.shape[0])

        boot_t = torch.tensor(boot, dtype=torch.float32)
        ci_low, ci_high = torch.quantile(boot_t, torch.tensor([0.025, 0.975])).tolist()

    return {
        "novel_fraction": float(novel_frac),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_candidates_geo": int(len(cand_idx)),
        "n_novel_func": int(n_novel),
        "base_mse": float(base_mse),
        "nmse": float(nmse),
        "explained_variance": float(explained_variance),
        "rms_act": float(rms_x),
        "rmse": float(rmse),
        "acts_mean_sq": float(acts_mean_sq),
        "acts_var": float(acts_var),
    }