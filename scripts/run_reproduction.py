
import gc
import json
import traceback
import torch
from sae_canonical_units.controls import random_decoder_control, shuffled_decoder_control
from sae_canonical_units.meta_sae import train_meta_sae
from sae_canonical_units.model_utils import get_activations, load_model
from sae_canonical_units.sae_loader import (
    load_gpt2_small_saes, load_gpt2_small_big,
    load_gemma_9b_saes,
    load_pythia_saes,
    load_llama_31_8b_saes,
    load_sae
)
from sae_canonical_units.stitching import stitching_novel_fraction

def run_one_width(acts, small, large, thresh, do_meta=True, do_control=True, bs=2048, n_bootstrap=1000):
    res = stitching_novel_fraction(small, large, acts, thresh=thresh, n_bootstrap=n_bootstrap)

    if do_meta:
        try:
            _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=1000, lr=3e-4, bs=bs)
            res["meta_sae_var_exp"] = float(var_exp)
        except Exception as e:
            print(f"meta_sae failed: {e}")
            res["meta_sae_var_exp"] = None

    if do_control:
        try:
            res["controls"] = {
                "random": random_decoder_control(small, large, acts, thresh=thresh),
                "shuffled": shuffled_decoder_control(small, large, acts, thresh=thresh)
            }
        except Exception as e:
            print(f"control failed: {e}")
            traceback.print_exc()
            res["controls"] = None

    return res

def run_gpt2_experiment():
    print("\n=== GPT2 Small (Local T4) ===")
    model = load_model("gpt2", use_remote=False)
    acts = get_activations(model, "gpt2", layer=8, hook_type="pre", n_tokens=8192, use_remote=False)
    del model
    torch.cuda.empty_cache(); gc.collect()

    results = []
    
    thresholds = [0.5, 0.6, 0.7, 0.8]
    for t in thresholds:
        print(f"--- GPT2 Threshold Sweep: 3k->12k t={t} ---")
        try:
            small, large = load_gpt2_small_saes()
            res = run_one_width(acts, small, large, thresh=t, do_meta=False, do_control=False, n_bootstrap=500)
            results.append({"experiment": "threshold_sweep", "threshold": t, "width": "3k->12k", "results": res})
            del small, large
        except Exception as e:
            print(f"Failed on threshold {t}: {e}")
        torch.cuda.empty_cache(); gc.collect()

    widths_to_test = [
        ("3k->12k", load_gpt2_small_saes, 0.7, True, True),
        ("3k->24k", load_gpt2_small_big, 0.7, True, False)
    ]

    for width, loader, thresh, do_meta, do_control in widths_to_test:
        print(f"--- GPT2 Width {width} ---")
        try:
            small, large = loader()
            res = run_one_width(acts, small, large, thresh=thresh, do_meta=do_meta, do_control=do_control)
            results.append({"width": width, "results": res})
            del small, large
        except Exception as e:
            print(f"Failed on width {width}: {e}")
            results.append({"width": width, "error": str(e)})
        torch.cuda.empty_cache(); gc.collect()

    del acts
    torch.cuda.empty_cache(); gc.collect()
    return results

def run_pythia_experiment():
    print("\n=== PYTHIA 70M (Local T4) ===")
    model = load_model("EleutherAI/pythia-70m-deduped", use_remote=False)
    acts = get_activations(model, "pythia", layer=3, hook_type="post", n_tokens=8192, use_remote=False)
    del model
    torch.cuda.empty_cache(); gc.collect()

    small, large = load_pythia_saes()
    res = run_one_width(acts, small, large, thresh=0.7, do_meta=True, do_control=True)

    print("--- Pythia Same-Width Control (Trainer 0 vs 1) ---")
    try:
        large_b = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_1")
        same_width_res = stitching_novel_fraction(large, large_b, acts, thresh=0.7, n_bootstrap=500)
        res["same_width_control"] = {
            "novel_fraction": same_width_res["novel_fraction"],
            "novel_frac_filtered": same_width_res.get("novel_frac_filtered", same_width_res["novel_fraction"]),
            "ci_low": same_width_res["ci_low"],
            "ci_high": same_width_res["ci_high"]
        }
        del large_b
    except Exception as e:
        print(f"Same-width control failed: {e}")

    del small, large, acts
    torch.cuda.empty_cache(); gc.collect()
    return [{"width": "16k", "results": res}]

def run_gemma_experiment():
    print("\n=== GEMMA 9B IT (Remote NDIF) ===")
    model = load_model("google/gemma-2-9b-it", use_remote=True)
    acts = get_activations(model, "gemma", layer=20, hook_type="post", n_tokens=8192, use_remote=True)
    del model
    torch.cuda.empty_cache(); gc.collect()

    results = []
    widths_to_test = [
        ("16k->131k", load_gemma_9b_saes, 0.7, True, False)  # FIX #2: was 0.4, now 0.7
    ]

    for width, loader, thresh, do_meta, do_control in widths_to_test:
        print(f"--- Gemma Width {width} t={thresh} ---")
        try:
            small, large = loader()
            res = run_one_width(acts, small, large, thresh=thresh, do_meta=do_meta, do_control=do_control, bs=4096, n_bootstrap=500)
            results.append({"width": width, "results": res})
            del small, large
        except Exception as e:
            print(f"Failed on width {width}: {e}")
            results.append({"width": width, "error": str(e)})
        torch.cuda.empty_cache(); gc.collect()

    del acts
    torch.cuda.empty_cache(); gc.collect()
    return results

def run_llama_experiment():
    print("\n=== LLAMA 3.1 8B (Remote NDIF) ===")
    model = load_model("meta-llama/Llama-3.1-8B", use_remote=True)
    acts = get_activations(model, "llama", layer=12, hook_type="post", n_tokens=8192, use_remote=True)
    del model
    torch.cuda.empty_cache(); gc.collect()

    small, large = load_llama_31_8b_saes()
    res = run_one_width(acts, small, large, thresh=0.7, do_meta=True, do_control=False, bs=4096, n_bootstrap=500)
    
    del small, large, acts
    torch.cuda.empty_cache(); gc.collect()
    return [{"width": "131k", "results": res}]

def main():
    all_results = {}
    for name, fn in [
        ("gpt2_small_L8", run_gpt2_experiment),
        ("pythia_70m_L3", run_pythia_experiment),
        ("gemma_9b_it_L20", run_gemma_experiment),
        # ("llama_31_8b_L12", run_llama_experiment)
    ]:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"{name} failed: {e}")
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\nSaved results.json")

if __name__ == "__main__":
    main()
