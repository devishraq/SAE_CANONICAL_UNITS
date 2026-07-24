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

# ===== PYTHIA-70M - L3 =====
def load_pythia_saes(): # 4k -> 16k = 4x
    small = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_0")
    large = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_2")
    return small, large

def load_pythia_big(): # 4k -> 32k = 8x
    small = load_sae("sae_bench_pythia70m_sweep_standard_ctx128_0712", "blocks.3.hook_resid_post__trainer_0")
    large = load_sae("pythia-70m-deduped-res-sm", "blocks.3.hook_resid_post")
    return small, large

# ===== GPT2-SMALL - L8 =====
def load_gpt2_small_saes(): # 3k -> 12k = 4x
    small = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_3072")
    large = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_12288")
    return small, large

def load_gpt2_small_big(): # 3k -> 24k = 8x 
    small = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_3072")
    large = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_24576")
    return small, large

def load_gpt2_small_98k(): # 3k -> 98k = 32x
    small = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_3072")
    large = load_sae("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_98304")
    return small, large

# ===== GEMMA-2-2B - L12 =====
def load_gemma_2b_saes(): # 16k -> 65k = 4x
    small = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")
    large = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_65k/canonical")
    return small, large

def load_gemma_2b_big(): # 16k -> 262k = 16x
    small = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")
    large = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_262k/canonical")
    return small, large

def load_gemma_2b_1m(): # 16k -> 1M = 64x
    small = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_16k/canonical")
    large = load_sae("gemma-scope-2b-pt-res-canonical", "layer_12/width_1m/canonical")
    return small, large

# ===== GEMMA-3-270M - L12 =====
def load_gemma3_270m_saes(): # 16k -> 65k = 4x
    small = load_sae("gemma-scope-2-270m-pt-res", "layer_12_width_16k_l0_medium")
    large = load_sae("gemma-scope-2-270m-pt-res", "layer_12_width_65k_l0_medium")
    return small, large

def load_gemma3_270m_big(): # 16k -> 262k = 16x
    small = load_sae("gemma-scope-2-270m-pt-res", "layer_12_width_16k_l0_medium")
    large = load_sae("gemma-scope-2-270m-pt-res", "layer_12_width_262k_l0_medium")
    return small, large

# ===== LLAMA-3.1-8B - L12 =====
def load_llama_31_8b_saes(): # 32k -> 131k      
    small = load_sae("llama_scope_lxr_8x", "l12r_8x")
    large = load_sae("llama_scope_lxr_32x", "l12r_32x")
    return small, large