import torch
from sae_lens import SAE

def load_sae(release, sae_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = out[0] if isinstance(out, tuple) else out
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae

def load_gpt2_small_saes():
    small_768 = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre_768")
    small_4k = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre_4096")
    large_16k = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre_16384")
    large_32k = load_sae("gpt2-small-res-jb", "blocks.8.hook_resid_pre_32768")
    return small_768, small_4k, large_16k, large_32k

def load_gemma_2b_saes():
    small = load_sae("gemma-scope-2b-pt-res", "layer_10/width_16k/canonical")
    large = load_sae("gemma-scope-2b-pt-res", "layer_10/width_65k/canonical")
    return small, large

def load_pythia_saes():
    small = load_sae("pythia-160m-deduped-res", "blocks.4.hook_resid_pre_4096")
    large = load_sae("pythia-160m-deduped-res", "blocks.4.hook_resid_pre_16384")
    return small, large