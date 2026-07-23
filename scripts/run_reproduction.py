import json
import traceback

import torch
from sae_canonical_units.controls import random_decoder_control
from sae_canonical_units.meta_sae import train_meta_sae
from sae_canonical_units.model_utils import get_activations, load_model
from sae_canonical_units.sae_loader import load_sae
from sae_canonical_units.stitching import stitching_novel_fraction

def run_gpt2_experiment():
    model = load_model("gpt2")
    acts = get_activations(model, layer=8, n_tokens=4096)
    
    results = []
    small = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre_4096")
    
    for w in ["16384", "32768"]:
        large = load_sae("gpt2-small-res-jb", f"blocks.8.hook_resid_pre_{w}")
        res = stitching_novel_fraction(small, large, acts, thresh=0.7, n_bootstrap=200)
        
        if w == "16384":
            _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=200)
            res["meta_sae_var_exp"] = var_exp
            res["controls"] = {"random": random_decoder_control(small, large, acts)}
            
        results.append({"width": w, "results": res})
        
    del model, acts, small, large
    torch.cuda.empty_cache()
    return results

def run_gemma_experiment():
    model = load_model("google/gemma-2-2b")
    acts = get_activations(model, layer=10, n_tokens=4096)
    
    results = []
    small = load_sae("gemma-scope-2b-pt-res", "layer_10/width_16k/canonical")
    large = load_sae("gemma-scope-2b-pt-res", "layer_10/width_65k/canonical")
    
    res = stitching_novel_fraction(small, large, acts, thresh=0.4, n_bootstrap=200)
    res["controls"] = {"random": random_decoder_control(small, large, acts)}
    
    _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=200, bs=4096)
    res["meta_sae_var_exp"] = var_exp
    results.append({"width": "65k", "results": res})
        
    del model, acts, small, large
    torch.cuda.empty_cache()
    return results

def run_pythia_experiment():
    model = load_model("EleutherAI/pythia-160m-deduped")
    acts = get_activations(model, layer=4, n_tokens=4096)
    
    results = []
    small = load_sae("pythia-160m-deduped-res", "blocks.4.hook_resid_pre_4096")
    large = load_sae("pythia-160m-deduped-res", "blocks.4.hook_resid_pre_16384")
    
    res = stitching_novel_fraction(small, large, acts, thresh=0.7, n_bootstrap=200)
    res["controls"] = {"random": random_decoder_control(small, large, acts)}
    
    _, var_exp = train_meta_sae(large, hidden=2048, k=4, epochs=200)
    res["meta_sae_var_exp"] = var_exp
    results.append({"width": "16k", "results": res})
        
    del model, acts, small, large
    torch.cuda.empty_cache()
    return results

def main():
    all_results = {}
    
    try:
        all_results["gpt2_small_L8"] = run_gpt2_experiment()
    except Exception as e:
        print(f"GPT-2 failed: {e}")
        traceback.print_exc()

    try:
        all_results["gemma_2b_L10"] = run_gemma_experiment()
    except Exception as e:
        print(f"Gemma failed: {e}")
        traceback.print_exc()

    try:
        all_results["pythia_160m_L4"] = run_pythia_experiment()
    except Exception as e:
        print(f"Pythia failed: {e}")
        traceback.print_exc()
    
    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()