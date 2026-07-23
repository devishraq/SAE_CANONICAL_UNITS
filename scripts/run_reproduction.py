import json
import traceback
import torch
from sae_canonical_units.controls import random_decoder_control
from sae_canonical_units.meta_sae import train_meta_sae
from sae_canonical_units.model_utils import get_activations, load_model
from sae_canonical_units.sae_loader import (
    load_gpt2_small_saes,
    load_gemma_2b_saes,
    load_pythia_saes,
)
from sae_canonical_units.stitching import stitching_novel_fraction

def run_one_width(acts, small, large, thresh, do_meta=True, do_control=True, bs=2048):
    res = stitching_novel_fraction(small, large, acts, thresh=thresh, n_bootstrap=200)

    if do_meta:
        _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=200, bs=bs)
        res["meta_sae_var_exp"] = var_exp
    if do_control:
        res["controls"] = {"random": random_decoder_control(small, large, acts)}

    return res

def run_gpt2_experiment():
    model = load_model("gpt2")
    acts = get_activations(model, layer=8, n_tokens=1024)

    small_768, small_4k, large_16k, large_32k = load_gpt2_small_saes()
    small = small_4k

    results = []
    results.append({
        "width": "16384",
        "results": run_one_width(acts, small, large_16k, thresh=0.7, do_meta=True, do_control=True)
    })

    results.append({
        "width": "32768",
        "results": run_one_width(acts, small, large_32k, thresh=0.7, do_meta=False, do_control=False)
    })

    del model, acts, small_768, small_4k, large_16k, large_32k
    torch.cuda.empty_cache()
    return results

def run_gemma_experiment():
    model = load_model("google/gemma-2-2b", use_remote=True)  
    acts = get_activations(model, layer=10, n_tokens=1024)

    small, large = load_gemma_2b_saes()
    res = run_one_width(acts, small, large, thresh=0.4, bs=4096)  

    del model, acts, small, large
    torch.cuda.empty_cache()
    return [{"width": "65k", "results": res}]

def run_pythia_experiment():
    model = load_model("EleutherAI/pythia-160m-deduped")
    acts = get_activations(model, layer=4, n_tokens=1024)

    small, large = load_pythia_saes()
    res = run_one_width(acts, small, large, thresh=0.7)

    del model, acts, small, large
    torch.cuda.empty_cache()
    return [{"width": "16k", "results": res}]

def main():
    all_results = {}

    for name, fn in [
        ("gpt2_small_L8", run_gpt2_experiment),
        ("gemma_2b_L10", run_gemma_experiment),
        ("pythia_160m_L4", run_pythia_experiment),
    ]:
        try:
            all_results[name] = fn()
        except Exception as e:
            print(f"{name} failed: {e}")
            traceback.print_exc()

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
        
    print("\nSaved results.json")

if __name__ == "__main__":
    main()