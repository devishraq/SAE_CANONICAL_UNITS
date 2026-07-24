# sae_canonical_units/sae_loader.py
import torch
from sae_lens import SAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sae(release, sae_id):
    print(f"Loading {release} / {sae_id}")
    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
    sae = out[0] if isinstance(out, tuple) else out

    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
        
    return sae

def load_gpt2_small_saes():
    release = "gpt2-small-res-jb-feature-splitting"
    small_768 = load_sae(release, "blocks.8.hook_resid_pre_768")
    small_4k = load_sae(release, "blocks.8.hook_resid_pre_3072")
    large_16k = load_sae(release, "blocks.8.hook_resid_pre_12288")
    large_32k = load_sae(release, "blocks.8.hook_resid_pre_24576")
    return small_768, small_4k, large_16k, large_32k

def load_gemma_2b_saes():
    release = "gemma-scope-2b-pt-res-canonical"
    small = load_sae(release, "layer_10/width_16k/canonical")
    large = load_sae(release, "layer_10/width_65k/canonical")
    return small, large

def load_pythia_saes():
    small = load_sae("pythia-70m-deduped-res-sm", "blocks.3.hook_resid_post")
    large = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_2") # 16384
    return small, large