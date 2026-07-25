import torch
import random
import numpy as np
import gc
from datasets import load_dataset
from transformer_lens import HookedTransformer
from nnsight import LanguageModel

HF_MAP = {"gpt2": "gpt2", "pythia": "EleutherAI/pythia-70m-deduped"}

def load_model(name="gpt2", use_remote=False):
    if use_remote:
        print(f"Loading {name} remote=True (NDIF)")
        return LanguageModel(name, remote=True)
    return None

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=8192, use_remote=False):
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    print(f"Extracting {n_tokens} {model_name} L{layer} {hook_type} remote={use_remote}")
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    if use_remote:
        tok = model.tokenizer
        raw = ds[0]["text"]

        for _ in range(200):
            raw += " " + ds[random.randint(0, 9999)]["text"]
            if len(tok(raw)["input_ids"]) > n_tokens + 512:
                break
        ids = tok(raw)["input_ids"][:n_tokens]
        all_acts = []

        BS = 128 if "gemma" in model_name.lower() else 256

        print(f"Using remote BS={BS} for {model_name}")

        for i in range(0, n_tokens, BS):
            batch = ids[i:i+BS]
            if not batch: break
            try:
                with model.trace(batch, remote=True):
                    resid = model.model.layers[layer].output[0].save()
                v = resid.value if hasattr(resid, 'value') else resid
                if v.dim()==3: v=v[0]
                all_acts.append(v.float().cpu())
            except Exception as e:
                print(f"Remote trace failed at {i}, retrying with BS=64: {e}")

                for j in range(i, min(i+BS, n_tokens), 64):
                    b2 = ids[j:j+64]
                    with model.trace(b2, remote=True):
                        resid = model.model.layers[layer].output[0].save()
                    v = resid.value if hasattr(resid, 'value') else resid
                    if v.dim()==3: v=v[0]
                    all_acts.append(v.float().cpu())
        return torch.cat(all_acts, dim=0)[:n_tokens]

    else:
        hf_id = HF_MAP.get(model_name, model_name)
        ht = HookedTransformer.from_pretrained(hf_id, device="cuda", dtype=torch.float16)
        raw = ds[0]["text"]
        while len(ht.tokenizer.encode(raw)) < n_tokens:
            raw += " " + ds[random.randint(0, 9999)]["text"]
        tokens = ht.to_tokens(raw, truncate=False)[:, :n_tokens]
        hook_name = f"blocks.{layer}.hook_resid_{hook_type}"
        all_acts = []
        for i in range(0, n_tokens, 512):
            batch = tokens[:, i:i+512]
            _, cache = ht.run_with_cache(batch)
            all_acts.append(cache[hook_name][0].float().cpu())
            del cache; torch.cuda.empty_cache()
        del ht;
        torch.cuda.empty_cache(); gc.collect()
        return torch.cat(all_acts, dim=0)