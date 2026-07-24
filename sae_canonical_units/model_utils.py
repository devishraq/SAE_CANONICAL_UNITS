import torch
import random
from datasets import load_dataset
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote} (float16)...")
    # Do NOT pass remote=True here. Pass it to model.trace() instead.
    return LanguageModel(name, device_map="auto", torch_dtype=torch.float16)

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=40960, batch_size=1024, use_remote=False):
    print(f"Extracting {n_tokens} activations at layer {layer} ({hook_type}) (batched)...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    
    random.seed(42)
    indices = random.sample(range(len(ds)), 50)
    raw_text = " ".join([ds[i]["text"] for i in indices])
    
    input_ids = model.tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    
    extract_layer = layer - 1 if hook_type == "pre" else layer
    
    all_acts = []
    for i in range(0, n_tokens, batch_size):
        batch = input_ids[i:i+batch_size].unsqueeze(0)
        if batch.shape[1] == 0: break
            
        # FIX: remote=True MUST go here, not in the model loader
        with model.trace(batch, remote=use_remote):
            if "gpt2" in model_name:
                resid = model.transformer.h[extract_layer].output[0].save()
            elif "pythia" in model_name:
                resid = model.gpt_neox.layers[extract_layer].output[0].save()
            else: # Llama & Gemma
                resid = model.model.layers[extract_layer].output[0].save()
        
        # FIX: Safely handle tensor vs proxy return
        if hasattr(resid, 'value'):
            acts = resid.value[0]
        else:
            acts = resid[0]
                
        acts = acts.float().cpu()
        all_acts.append(acts)
        
    acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"Extracted shape: {acts.shape}")
    return acts