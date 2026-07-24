import torch
from datasets import load_dataset
from nnterp import StandardizedTransformer

def load_model(name="gpt2"):
    print(f"Loading {name}...")
    return StandardizedTransformer(name, device_map="auto", torch_dtype=torch.float16)

def get_activations(model, layer=8, n_tokens=1024):
    print(f"Extracting activations at layer {layer}...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    raw_text = " ".join([ds[i]["text"] for i in range(5)])
    enc = model.tokenizer(raw_text, truncation=True, max_length=n_tokens, return_tensors="pt")
    text = model.tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with model.trace(text):
                resid = model.layers_output[layer].save()

    acts = resid.value if hasattr(resid, "value") else resid
    if acts.dim() == 3:
        acts = acts[0]
    acts = acts.reshape(-1, acts.shape[-1])[:n_tokens].float().detach().cpu()
    print(f"Extracted shape: {acts.shape}")
    return acts