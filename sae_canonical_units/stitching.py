import torch
import torch.nn.functional as F
import gc

def stitching_novel_fraction(sae_small, sae_large, acts, thresh=0.7, n_bootstrap=500, min_improvement=0.0):
    device = sae_small.W_dec.device
    acts = acts.to(device).float()

    with torch.no_grad():
        # ---- small SAE recon ----
        z_s = sae_small.encode(acts)
        if isinstance(z_s, tuple): z_s = z_s[0]
        z_s = z_s.float()
        x_s = sae_small.decode(z_s).float()
        E = acts - x_s

        base_mse = (E**2).mean().item()
        acts_var = acts.var().item()
        explained_variance = 1.0 - base_mse / (acts_var + 1e-8)
        if explained_variance < 0:
            raise ValueError(f"SAE/activation mismatch EV={explained_variance}. Check model hooks.")

        # ---- cosine ----
        small_n = F.normalize(sae_small.W_dec.float(), dim=1)
        large_n = F.normalize(sae_large.W_dec.float(), dim=1)
        n_large = large_n.shape[0]
        n_small = small_n.shape[0]

        max_sim = torch.empty(n_large, device=device)
        # if 131k x 16k = 2.1B entries -> do on CPU
        CHUNK = 2048
        if n_large * n_small > 50_000_000:
            small_cpu = small_n.float().cpu()
            for i in range(0, n_large, CHUNK):
                lc = large_n[i:i+CHUNK].float().cpu()
                sim = lc @ small_cpu.T
                max_sim[i:i+CHUNK] = sim.max(dim=1).values.to(device)
                del sim, lc
        else:
            for i in range(0, n_large, CHUNK):
                sim = large_n[i:i+CHUNK] @ small_n.T
                max_sim[i:i+CHUNK] = sim.max(dim=1).values
        del small_n, large_n

        cand_idx = torch.where(max_sim < thresh)[0]
        if len(cand_idx)==0:
            return {"novel_fraction":0.0,"novel_frac_filtered":0.0,"ci_low":0.0,"ci_high":0.0,
                    "n_candidates_geo":0,"n_novel_func":0,"base_mse":base_mse,"explained_variance":explained_variance}

        # ---- firing counts batched to avoid 8192x131k = 1B OOM ----
        firing_counts = torch.zeros(n_large, device=device)
        tok_bs = 512
        for i in range(0, acts.shape[0], tok_bs):
            ab = acts[i:i+tok_bs]
            zb = sae_large.encode(ab)
            if isinstance(zb, tuple): zb = zb[0]
            firing_counts += (zb.float() > 0).sum(0)
            del zb, ab
        active_mask = firing_counts > 20
        cand_idx = cand_idx[active_mask[cand_idx]]
        if len(cand_idx)==0:
            return {"novel_fraction":0.0,"novel_frac_filtered":0.0,"ci_low":0.0,"ci_high":0.0,
                    "n_candidates_geo":0,"n_novel_func":0,"base_mse":base_mse,"explained_variance":explained_variance}

        # ---- chunked dot product to avoid 8192x50k = 400M OOM ----
        n_cand = len(cand_idx)
        dec_cand_all = sae_large.W_dec[cand_idx].float() # (n_cand, d)
        dec_norm2_all = (dec_cand_all**2).sum(1) # (n_cand)

        dot_EC_all = torch.zeros(n_cand, device=device)
        z_norm2_all = torch.zeros(n_cand, device=device)

        # For bootstrap we need per-token contributions
        N = acts.shape[0]
        # Reduce bootstrap for huge width to fit T4 time
        if n_cand > 20000:
            n_bootstrap = min(n_bootstrap, 100)

        CAND_CHUNK = 4096
        # we will store M_dot per chunk on CPU for bootstrap
        boot_counts = []

        for c_start in range(0, n_cand, CAND_CHUNK):
            c_end = min(c_start+CAND_CHUNK, n_cand)
            dec_chunk = dec_cand_all[c_start:c_end].to(device) # 4096 x d
            dec_norm_chunk = dec_norm2_all[c_start:c_end].to(device)
            chunk_len = c_end - c_start

            dot_chunk = torch.zeros(chunk_len, device=device)
            znorm_chunk = torch.zeros(chunk_len, device=device)

            # M for bootstrap: (N, chunk_len) on CPU to save VRAM
            M_dot_cpu = torch.zeros(N, chunk_len, dtype=torch.float32)
            M_norm_cpu = torch.zeros(N, chunk_len, dtype=torch.float32)

            for t_start in range(0, N, tok_bs):
                t_end = min(t_start+tok_bs, N)
                ab = acts[t_start:t_end]
                Eb = E[t_start:t_end]
                zb = sae_large.encode(ab)
                if isinstance(zb, tuple): zb = zb[0]
                zb = zb[:, cand_idx[c_start:c_end]].float() # (tok_bs, chunk)

                Edot = Eb @ dec_chunk.T # (tok_bs, chunk)
                dot = zb * Edot
                dot_chunk += dot.sum(0)
                znorm_chunk += (zb**2).sum(0)

                M_dot_cpu[t_start:t_end] = dot.float().cpu()
                M_norm_cpu[t_start:t_end] = (zb**2).float().cpu() * dec_norm_chunk.float().cpu()
                del zb, Edot, dot, ab, Eb

            dot_EC_all[c_start:c_end] = dot_chunk
            z_norm2_all[c_start:c_end] = znorm_chunk

            # bootstrap for this chunk
            norm_chunk = znorm_chunk * dec_norm_chunk # = z_norm2 * dec_norm2
            if n_bootstrap>0:
                boot_chunk = []
                for _ in range(n_bootstrap):
                    idx = torch.randint(0, N, (N,))
                    d = M_dot_cpu[idx].sum(0)
                    nrm = M_norm_cpu[idx].sum(0)
                    boot_chunk.append(((2*d - nrm) > min_improvement).float())
                # (n_boot, chunk)
                boot_counts.append(torch.stack(boot_chunk, dim=0))
            del M_dot_cpu, M_norm_cpu, dec_chunk

        norm_C2_all = z_norm2_all * dec_norm2_all
        improves = (2*dot_EC_all - norm_C2_all) > min_improvement
        n_novel = improves.sum().item()

        # bootstrap CI on filtered fraction
        if n_bootstrap>0 and boot_counts:
            boot_all = torch.cat(boot_counts, dim=1) # (n_boot, n_cand)
            boot_frac = boot_all.sum(1) / n_cand
            ci_low, ci_high = torch.quantile(boot_frac, torch.tensor([0.025,0.975])).tolist()
        else:
            ci_low, ci_high = 0.0, 0.0

        return {
            "novel_fraction": float(n_novel / sae_large.W_dec.shape[0]),
            "novel_frac_filtered": float(n_novel / max(1,n_cand)),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "n_candidates_geo": int(n_cand),
            "n_novel_func": int(n_novel),
            "base_mse": float(base_mse),
            "explained_variance": float(explained_variance),
        }