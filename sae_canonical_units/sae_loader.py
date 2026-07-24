import torch
from sae_lens import SAE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_sae(release, sae_id):
    print(f"Loading {release} / {sae_id}")
    loaded = SAE.from_pretrained(release=release, sae_id=sae_id, device=str(DEVICE))
    sae = loaded[0] if isinstance(loaded, tuple) else getattr(loaded, 'sae', loaded)
    sae = sae.to(DEVICE).to(torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    # FIX: don't assert hook_name - different SAE families use different field names
    hook = getattr(sae.cfg, 'hook_name', None) or getattr(sae.cfg, 'hook_point', None)
    return sae