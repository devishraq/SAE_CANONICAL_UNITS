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
    small = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_3072")
    large = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_12288")
    return small, large

def load_gpt2_small_big():
    small = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_3072")
    large = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_24576")
    return small, large

def load_pythia_saes():
    small = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_0")
    large = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_2")
    return small, large

def load_gemma_9b_saes():
    small = load_sae("gemma-scope-9b-pt-res-canonical", "layer_20/width_16k/canonical")
    large = load_sae("gemma-scope-9b-pt-res-canonical", "layer_20/width_131k/canonical")
    return small, large

def load_llama_31_8b_saes():
    small = load_sae("llama_scope_lxr_8x", "l12r_8x")
    large = load_sae("llama_scope_lxr_32x", "l12r_32x")
    return small, large