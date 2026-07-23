import torch
from datasets import load_dataset
from nnterp import StandardizedTransformer

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} (float16)...")

    if use_remote:
        return StandardizedTransformer(name, remote=True, torch_dtype=torch.float16)
    else:
        return StandardizedTransformer(name, torch_dtype=torch.float16, device_map="auto")

def get_activations(model, layer=8, n_tokens=1024):
    print(f"Extracting activations at layer {layer}...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    text = " ".join([ds[i]["text"] for i in range(3)])[:15000]

    with model.trace(text):
        resid = model.layers_output[layer].save()

    acts = resid.value
    if acts.dim() == 3:
        acts = acts[0]
    acts = acts.reshape(-1, acts.shape[-1])[:n_tokens].float().detach()
    if torch.cuda.is_available():
        acts = acts.to("cuda")

    print(f"Extracted shape: {acts.shape}")
    return acts