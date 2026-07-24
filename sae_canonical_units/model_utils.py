import torch
import random
from datasets import load_dataset
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote} (float16)...")
    if use_remote:
        return LanguageModel(name, remote=True, torch_dtype=torch.float16)
    else:
        return LanguageModel(name, device_map="auto", torch_dtype=torch.float16)

def get_activations(model, model_name, layer, n_tokens=51200, batch_size=1024):
    print(f"Extracting {n_tokens} activations at layer {layer} (batched)...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    
    random.seed(42)
    indices = random.sample(range(len(ds)), 50)
    raw_text = " ".join([ds[i]["text"] for i in indices])
    
    input_ids = model.tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    
    all_acts = []
    for i in range(0, n_tokens, batch_size):
        batch = input_ids[i:i+batch_size].unsqueeze(0)
        if batch.shape[1] == 0: break
            
        with model.trace(batch):
            # Native HuggingFace paths! No nnterp standardization!
            if "gpt2" in model_name:
                resid = model.transformer.h[layer].output[0].save()
            elif "pythia" in model_name:
                resid = model.gpt_neox.layers[layer].output[0].save()
            else: # Llama & Gemma
                resid = model.model.layers[layer].output[0].save()
                
        # Move to CPU immediately to save VRAM, cast to float32 for math safety
        acts = resid.value[0].float().cpu()
        all_acts.append(acts)
        
    acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"Extracted shape: {acts.shape}")
    return acts