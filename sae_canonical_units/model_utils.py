import torch, random, gc
from datasets import load_dataset
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote}")
    m = LanguageModel(name, device_map="cuda", torch_dtype=torch.float16)
    m.model.config.use_cache = False
    return m

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=40960, batch_size=256, use_remote=False):
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    # FIX: generate enough raw text - 50 docs is only ~20k tokens
    raw_text = ""
    tok = model.tokenizer
    while len(tok(raw_text)["input_ids"]) < n_tokens + 2048:
        raw_text += " " + ds[random.randint(0, len(ds)-1)]["text"]

    input_ids = tok(raw_text, return_tensors="pt")["input_ids"][0]
    extract_layer = layer - 1 if hook_type == "pre" else layer
    all_acts = []

    for i in range(0, n_tokens, batch_size):
        batch = input_ids[i:i+batch_size].unsqueeze(0)
        if batch.shape[1]==0: break
        with torch.no_grad():
            with model.trace(batch, remote=use_remote):
                if "gpt2" in model_name:
                    resid = model.transformer.h[extract_layer].output[0].save()
                elif "pythia" in model_name:
                    resid = model.gpt_neox.layers[extract_layer].output[0].save()
                else:
                    resid = model.model.layers[extract_layer].output[0].save()

        acts = resid.value[0].float().cpu() if hasattr(resid,'value') else resid[0].float().cpu()
        all_acts.append(acts)
        del resid, batch
        torch.cuda.empty_cache()
        gc.collect()

    return torch.cat(all_acts, dim=0)[:n_tokens]