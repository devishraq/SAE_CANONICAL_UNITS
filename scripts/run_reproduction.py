import gc
import json
import traceback
import torch
from sae_canonical_units.controls import random_decoder_control
from sae_canonical_units.meta_sae import train_meta_sae
from sae_canonical_units.model_utils import get_activations, load_model
from sae_canonical_units.sae_loader import (
    load_gpt2_small_saes, load_gpt2_small_big, load_gpt2_small_98k,
    load_gemma_2b_saes, load_gemma_2b_big, load_gemma_2b_1m,
    load_pythia_saes,
    load_llama_31_8b_saes
)
from sae_canonical_units.stitching import stitching_novel_fraction

def run_one_width(acts, small, large, thresh, do_meta=True, do_control=True, bs=2048):
    res = stitching_novel_fraction(small, large, acts, thresh=thresh, n_bootstrap=200)

    if do_meta:
        try:
            _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=200, bs=bs)
            res["meta_sae_var_exp"] = float(var_exp)
        except Exception as e:
            print(f"meta_sae failed: {e}")
            res["meta_sae_var_exp"] = None

    if do_control:
        try:
            res["controls"] = {"random": random_decoder_control(small, large, acts)}
        except Exception as e:
            print(f"control failed: {e}")
            res["controls"] = None

    return res

def run_gpt2_experiment():
    print("\n=== GPT2 Small (Width Curve) ===")
    model = load_model("gpt2")
    acts = get_activations(model, layer=8, n_tokens=1024)
    del model
    torch.cuda.empty_cache(); gc.collect()

    results = []
    
    # (Label, Loader Function, thresh, do_meta, do_control)
    widths_to_test = [
        ("12288", load_gpt2_small_saes, 0.7, True, True),
        ("24576", load_gpt2_small_big, 0.7, False, False),
        ("98304", load_gpt2_small_98k, 0.7, False, False) 
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

def run_gemma_experiment():
    print("\n=== GEMMA 2B (Width Curve) ===")
    model = load_model("google/gemma-2-2b")
    acts = get_activations(model, layer=10, n_tokens=1024)
    del model
    torch.cuda.empty_cache(); gc.collect()

    results = []
    
    widths_to_test = [
        ("65k", load_gemma_2b_saes, 0.4, False, False),
        ("262k", load_gemma_2b_big, 0.4, False, False),
        ("1m", load_gemma_2b_1m, 0.4, False, False)
    ]

    for width, loader, thresh, do_meta, do_control in widths_to_test:
        print(f"--- Gemma Width {width} ---")
        try:
            small, large = loader()
            res = run_one_width(acts, small, large, thresh=thresh, do_meta=do_meta, do_control=do_control, bs=4096)
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
    print("\n=== PYTHIA 70M ===")
    model = load_model("EleutherAI/pythia-70m-deduped")
    acts = get_activations(model, layer=3, n_tokens=1024)
    del model
    torch.cuda.empty_cache(); gc.collect()

    small, large = load_pythia_saes()
    res = run_one_width(acts, small, large, thresh=0.7, do_meta=True, do_control=True)

    del small, large, acts
    torch.cuda.empty_cache(); gc.collect()
    return [{"width": "16k", "results": res}]

def run_llama_experiment():
    print("\n=== LLAMA 3.1 8B (Remote NDIF) ===")
    model = load_model("meta-llama/Llama-3.1-8B", use_remote=True)
    acts = get_activations(model, layer=12, n_tokens=1024)
    
    del model
    torch.cuda.empty_cache(); gc.collect()

    small, large = load_llama_31_8b_saes()
    # 131k width SAE is huge, keep batch size high and skip meta-sae to be safe
    res = run_one_width(acts, small, large, thresh=0.7, do_meta=False, do_control=False, bs=4096)
    
    del small, large, acts
    torch.cuda.empty_cache(); gc.collect()
    return [{"width": "131k", "results": res}]

def main():
    all_results = {}
    for name, fn in [
        ("gpt2_small_L8", run_gpt2_experiment),
        ("gemma_2b_L10", run_gemma_experiment),
        ("pythia_70m_L3", run_pythia_experiment),
        ("llama_31_8b_L12", run_llama_experiment),
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