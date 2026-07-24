import torch
import random
from datasets import load_dataset
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote} (float16)...")
    if use_remote:
        # Raw nnsight for remote NDIF models
        return LanguageModel(name, remote=True)
    else:
        # TransformerLens for local models to fold LayerNorms perfectly!
        return LanguageModel(name, device_map="auto", torch_dtype=torch.float16, transformer_lens=True)

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=8192, batch_size=1024, use_remote=False):
    print(f"Extracting {n_tokens} activations at layer {layer} ({hook_type}) (batched)...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    
    random.seed(42)
    indices = random.sample(range(len(ds)), 20)
    raw_text = " ".join([ds[i]["text"] for i in indices])
    
    input_ids = model.tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    
    all_acts = []
    for i in range(0, n_tokens, batch_size):
        batch = input_ids[i:i+batch_size].unsqueeze(0)
        if batch.shape[1] == 0: break
        
        with model.trace(batch, remote=use_remote):
            if use_remote:
                # Raw HuggingFace path for remote Llama/Gemma
                resid = model.model.layers[layer].output[0].save()
            else:
                # TransformerLens path for local GPT-2/Pythia
                if hook_type == "pre":
                    resid = model.transformer.h[layer].hook_resid_pre.save()
                else:
                    resid = model.transformer.h[layer].hook_resid_post.save()
        
        acts_tensor = resid.value if hasattr(resid, 'value') else resid
        
        if acts_tensor.dim() == 3:
            acts_tensor = acts_tensor[0]
            
        acts_tensor = acts_tensor.reshape(-1, acts_tensor.shape[-1]).float().cpu()
        all_acts.append(acts_tensor)
        
    acts = torch.cat(all_acts, dim=0)[:n_tokens]
    print(f"Extracted shape: {acts.shape}")
    return acts