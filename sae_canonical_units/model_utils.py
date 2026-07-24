import torch
import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    if use_remote:
        print(f"Loading {name} remote=True (NDIF)")
        return LanguageModel(name, remote=True)
    return None

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=8192, use_remote=False):
    print(f"Extracting {n_tokens} activations for {model_name}...")
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    random.seed(42)
    indices = random.sample(range(len(ds)), 20)
    raw_text = " ".join([ds[i]["text"] for i in indices])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    input_ids = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
    
    if use_remote:
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
            all_acts.append(acts_tensor.float().cpu())
        acts = torch.cat(all_acts, dim=0)[:n_tokens]
        del model
        torch.cuda.empty_cache()
        return acts
    else:
        # Local path using HuggingFace directly (Bulletproof!)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
        hf_model.eval()
        
        # GPT2 has 1024 context, Pythia has 2048. Batch size 512 guarantees it fits.
        batch_size = 512
        all_acts = []
        
        for i in range(0, n_tokens, batch_size):
            batch = input_ids[i:i+batch_size].unsqueeze(0).to("cuda")
            if batch.shape[1] == 0: break
            
            with torch.no_grad():
                outputs = hf_model(batch, output_hidden_states=True)
            
            # hidden_states tuple: 0=embed, 1=output block 0, ...
            # resid_pre_N == output of block N-1 == hidden_states[N]
            # resid_post_N == output of block N == hidden_states[N+1]
            hs_idx = layer if hook_type == "pre" else layer + 1
            acts = outputs.hidden_states[hs_idx][0].float().cpu()
            all_acts.append(acts)
            
        del hf_model
        torch.cuda.empty_cache()
        return torch.cat(all_acts, dim=0)[:n_tokens]