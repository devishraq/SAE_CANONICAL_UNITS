import torch, random, gc
from datasets import load_dataset
from nnsight import LanguageModel

def load_model(name="gpt2", use_remote=False):
    print(f"Loading {name} remote={use_remote}")
    if use_remote:
        # remote=True must NOT have device_map / dtype
        return LanguageModel(name, remote=True)
    else:
        # local - cuda + fp16 to fit 14GB
        model = LanguageModel(name, device_map="cuda", torch_dtype=torch.float16)
        # optional, don't crash if config path differs
        try:
            model._model.config.use_cache = False
        except:
            pass
        return model

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=40960, batch_size=256, use_remote=False):
    print(f"Extracting {n_tokens} at layer {layer} ({hook_type}) bs={batch_size}")
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    # FIX: generate ENOUGH tokens - 50 docs is only ~15k
    raw_text = ""
    while len(model.tokenizer(raw_text)["input_ids"]) < n_tokens + 2048:
        raw_text += " " + ds[random.randint(0, len(ds)-1)]["text"]

    input_ids = model.tokenizer(raw_text, return_tensors="pt")["input_ids"][0]
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