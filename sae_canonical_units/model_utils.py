import torch
from datasets import load_dataset
from nnterp import StandardizedTransformer

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote}...")
    if use_remote:
        return StandardizedTransformer(name, remote=True)
    else:
        return StandardizedTransformer(name, device_map="auto", torch_dtype=torch.float16)

def get_activations(model, layer=8, n_tokens=1024, use_remote=False):
    print(f"Extracting activations at layer {layer}...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    raw_text = " ".join([ds[i]["text"] for i in range(5)])
    
    enc = model.tokenizer(raw_text, truncation=True, max_length=n_tokens, return_tensors="pt")
    
    if not use_remote:
        inputs = {k: v.to("cuda") for k, v in enc.items()}
    else:
        inputs = enc

    with torch.no_grad():
        with model.trace(inputs, remote=use_remote):
            resid = resid = model.layers_output[layer].save()

    acts = resid.value if hasattr(resid, "value") else resid
    if hasattr(acts, "value"):
        acts = acts.value

    if acts.dim() == 3:
        acts = acts[0]
        
    acts = acts.reshape(-1, acts.shape[-1])[:n_tokens].float().detach()
    print(f"Extracted shape: {acts.shape}")
    return acts.cpu()