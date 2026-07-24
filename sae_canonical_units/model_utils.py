import torch
from datasets import load_dataset
from nnterp import StandardizedTransformer

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} (float16) remote={use_remote}...")
    if use_remote:
        return StandardizedTransformer(name, remote=True, torch_dtype=torch.float16)
    else:
        return StandardizedTransformer(name, device_map="auto", torch_dtype=torch.float16)

def get_activations(model, layer=8, n_tokens=1024):
    print(f"Extracting activations at layer {layer}...")

    ds = load_dataset("NeelNanda/pile-10k", split="train")
    raw_text = " ".join([ds[i]["text"] for i in range(5)])
    enc = model.tokenizer(raw_text, truncation=True, max_length=n_tokens, return_tensors="pt")
    truncated_text = model.tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)

    with model.trace(truncated_text):
        resid = model.layers_output[layer].save()

    acts = resid.value if hasattr(resid, "value") else resid

    if hasattr(acts, "value"):
        acts = acts.value

    if acts.dim() == 3:
        acts = acts[0]
    acts = acts.reshape(-1, acts.shape[-1])[:n_tokens].float().detach()

    if torch.cuda.is_available():
        acts = acts.to("cuda")

    print(f"Extracted shape: {acts.shape}")
    return acts