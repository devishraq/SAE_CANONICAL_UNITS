import torch
from datasets import load_dataset
from nnterp import StandardizedTransformer

def load_model(name="gpt2", use_remote=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {name} on {device}...")
    return StandardizedTransformer(name, device=device)

def get_activations(model, layer=8, n_tokens=1024):
    print(f"Extracting activations at layer {layer}...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    text = " ".join([ds[i]["text"] for i in range(3)])[:15000]

    inputs = model.tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(model.device)

    with model.trace(inputs):
        resid = model.layers[layer].output[0].save()

    acts = resid.value[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    acts = acts.reshape(-1, acts.shape[-1])[:n_tokens].detach().to(device)
    print(f"Extracted activations shape: {acts.shape}")
    return acts