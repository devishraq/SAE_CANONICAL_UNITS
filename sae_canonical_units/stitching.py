import torch
import torch.nn.functional as F

def stitching_novel_fraction(sae_small, sae_large, acts, thresh=0.7, n_bootstrap=500, min_improvement=0.0):
    device = sae_small.W_dec.device
    acts = acts.to(device).float() # MUST be float32 for stable SAE encoding
    
    with torch.no_grad():
        z_s = sae_small.encode(acts)
        x_s = sae_small.decode(z_s)
        
        E = (acts - x_s).float()
        acts_f = acts.float()
        
        del z_s, x_s
        torch.cuda.empty_cache()
        
        base_mse = (E**2).mean().item()
        acts_var = acts_f.var().item()
        explained_variance = 1.0 - base_mse / (acts_var + 1e-8)
        
        if explained_variance < 0:
            raise ValueError(f"SAE/activation mismatch EV={explained_variance}. Check model hooks.")

        small_n = F.normalize(sae_small.W_dec.float(), dim=1)
        large_n = F.normalize(sae_large.W_dec.float(), dim=1)
        
        max_sim = torch.empty(large_n.shape[0], device=device)
        chunk_size = 8192
        for i in range(0, large_n.shape[0], chunk_size):
            sim_chunk = large_n[i:i+chunk_size] @ small_n.T
            max_sim[i:i+chunk_size] = sim_chunk.max(dim=1).values
            
        cand_idx = torch.where(max_sim < thresh)[0]

        z_l = sae_large.encode(acts_f.to(sae_large.W_dec.dtype)).float() # Encode in native dtype, return to float32
        del acts_f
        torch.cuda.empty_cache()
        
        firing_counts = (z_l > 0).sum(0)
        active_mask = firing_counts > 20
        valid_cand_mask = active_mask[cand_idx]
        cand_idx = cand_idx[valid_cand_mask]

        if len(cand_idx) == 0:
            return {
                "novel_fraction": 0.0, "novel_frac_filtered": 0.0,
                "ci_low": 0.0, "ci_high": 0.0,
                "n_candidates_geo": 0, "n_novel_func": 0,
                "base_mse": base_mse, "explained_variance": explained_variance
            }

        z_cand = z_l[:, cand_idx].float()
        del z_l
        torch.cuda.empty_cache()
        
        dec_cand = sae_large.W_dec[cand_idx].float()

        z_norm2 = (z_cand**2).sum(0)
        dec_norm2 = (dec_cand**2).sum(1)
        norm_C2 = z_norm2 * dec_norm2
        
        dot_EC = ((z_cand.T @ E) * dec_cand).sum(1)
        improves = (2 * dot_EC - norm_C2) > min_improvement

        n_novel = improves.sum().item()
        novel_frac_raw = n_novel / sae_large.W_dec.shape[0]
        novel_frac_filtered = n_novel / len(cand_idx)

        N = E.shape[0]
        boot = []
        for _ in range(n_bootstrap):
            idx = torch.randint(0, N, (N,), device=device)
            Eb = E[idx]
            z_cb = z_cand[idx]
            
            dot_b = ((z_cb.T @ Eb) * dec_cand).sum(1)
            norm_b = (z_cb**2).sum(0) * dec_norm2
            boot.append(((2*dot_b - norm_b) > min_improvement).sum().item() / len(cand_idx))

        boot_t = torch.tensor(boot, dtype=torch.float32)
        ci_low, ci_high = torch.quantile(boot_t, torch.tensor([0.025, 0.975])).tolist()

    return {
        "novel_fraction": float(novel_frac_raw),
        "novel_frac_filtered": float(novel_frac_filtered),
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_candidates_geo": int(len(cand_idx)),
        "n_novel_func": int(n_novel),
        "base_mse": float(base_mse),
        "explained_variance": float(explained_variance),
    }