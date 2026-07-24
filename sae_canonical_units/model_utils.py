import os
# Prevent CUDA OOM from memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import random
from datasets import load_dataset
from transformer_lens import HookedTransformer
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    if use_remote:
        print(f"Loading {name} remote=True (NDIF)")
        return LanguageModel(name, remote=True)
    print(f"Local model {name} will be handled by TransformerLens.")
    return None

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=4096, use_remote=False):
    print(f"Extracting {n_tokens} activations at layer {layer} ({hook_type})...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    
    random.seed(42)
    indices = random.sample(range(len(ds)), 20)
    raw_text = " ".join([ds[i]["text"] for i in indices])
    
    if use_remote:
        # Remote path for Llama/Gemma via nnsight
        input_ids = model.tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
        batch_size = 1024
        all_acts = []
        for i in range(0, n_tokens, batch_size):
            batch = input_ids[i:i+batch_size].unsqueeze(0)
            if batch.shape[1] == 0: break
            with model.trace(batch, remote=True):
                resid = model.model.layers[layer].output[0].save()
            acts_tensor = resid.value if hasattr(resid, 'value') else resid
            if acts_tensor.dim() == 3:
                acts_tensor = acts_tensor[0]
            acts_tensor = acts_tensor.reshape(-1, acts_tensor.shape[-1]).float().cpu()
            all_acts.append(acts_tensor)
        acts = torch.cat(all_acts, dim=0)[:n_tokens]
        del model
        torch.cuda.empty_cache()
        return acts
    else:
        # Local path using TransformerLens (Perfect for SAELens!)
        tl_model = HookedTransformer.from_pretrained(model_name, device="cuda")
        tokens = tl_model.to_tokens(raw_text, truncate=False)[0, :n_tokens].unsqueeze(0)
        
        hook_name = f"blocks.{layer}.hook_resid_{hook_type}"
        _, cache = tl_model.run_with_cache(tokens, names_filter=hook_name)
        acts = cache[hook_name][0].float().cpu()
        
        del tl_model
        torch.cuda.empty_cache()
        return acts