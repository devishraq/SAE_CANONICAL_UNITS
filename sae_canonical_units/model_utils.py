import torch, random, gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

HF_MAP = {
    "gpt2": "gpt2",
    "pythia": "EleutherAI/pythia-70m-deduped",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/Llama-3.1-8B",
}

def load_model(name="gpt2", use_remote=False):
    if use_remote:
        print(f"Loading {name} remote=True")
        return LanguageModel(name, remote=True)
    # local: we use HF directly to avoid nnsight OOM, so return None
    return None

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=8192, use_remote=False):
    hf_id = HF_MAP.get(model_name, model_name)
    print(f"Extracting {n_tokens} for {hf_id} layer {layer} {hook_type} remote={use_remote}")
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    # Build ENOUGH text
    tokenizer = AutoTokenizer.from_pretrained(hf_id) if not use_remote else model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_text = ""
    while len(tokenizer(raw_text)["input_ids"]) < n_tokens + 1024:
        raw_text += " " + ds[random.randint(0, len(ds)-1)]["text"]

    input_ids = tokenizer(raw_text, return_tensors="pt")["input_ids"][0]

    if use_remote:
        all_acts = []
        for i in range(0, n_tokens, 1024):
            batch = input_ids[i:i+1024].unsqueeze(0)
            if batch.shape[1]==0: break
            with model.trace(batch, remote=True):
                resid = model.model.layers[layer].output[0].save()
            v = resid.value if hasattr(resid,'value') else resid
            if v.dim()==3: v=v[0]
            all_acts.append(v.float().cpu())
            del resid
        return torch.cat(all_acts)[:n_tokens]
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, device_map="cuda")
        hf_model.eval()
        all_acts = []
        for i in range(0, n_tokens, 128): # 128 not 512/1024 - fits T4
            batch = input_ids[i:i+128].unsqueeze(0).to("cuda")
            if batch.shape[1]==0: break
            with torch.no_grad():
                out = hf_model(batch, output_hidden_states=True)
            idx = layer if hook_type=="pre" else layer+1
            all_acts.append(out.hidden_states[idx][0].float().cpu())
            del out, batch
            torch.cuda.empty_cache()
        del hf_model
        torch.cuda.empty_cache()
        gc.collect()
        return torch.cat(all_acts)[:n_tokens]