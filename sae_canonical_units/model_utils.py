import torch, random, gc
from datasets import load_dataset
from transformer_lens import HookedTransformer
from nnsight import LanguageModel

HF_MAP = {"gpt2":"gpt2", "pythia":"EleutherAI/pythia-70m-deduped"}

def load_model(name="gpt2", use_remote=False):
    if use_remote:
        return LanguageModel(name, remote=True)
    return None # local uses HookedTransformer inside get_activations

def get_activations(model, model_name, layer, hook_type="pre", n_tokens=8192, use_remote=False):
    print(f"Extracting {n_tokens} {model_name} L{layer} {hook_type} remote={use_remote}")
    ds = load_dataset("NeelNanda/pile-10k", split="train")

    if use_remote:
        # NDIF path - keep your working code
        tok = model.tokenizer
        raw = ""
        while len(tok(raw)["input_ids"]) < n_tokens+1024:
            raw += " " + ds[random.randint(0,len(ds)-1)]["text"]
        ids = tok(raw)["input_ids"]
        all_acts=[]
        for i in range(0,n_tokens,1024):
            batch = ids[i:i+1024]
            if not batch: break
            with model.trace(batch, remote=True):
                resid = model.model.layers[layer].output[0].save()
            v = resid.value if hasattr(resid,'value') else resid
            if v.dim()==3: v=v[0]
            all_acts.append(v.float().cpu())
        return torch.cat(all_acts)[:n_tokens]
    else:
        # LOCAL: HookedTransformer = exact SAEs training distribution = EV > 0
        hf_id = HF_MAP.get(model_name, model_name)
        ht = HookedTransformer.from_pretrained(hf_id, device="cuda", dtype=torch.float16)
        raw = ""
        while ht.to_tokens(raw).shape[1] < n_tokens+512:
            raw += " " + ds[random.randint(0,len(ds)-1)]["text"]
        tokens = ht.to_tokens(raw)[:, :n_tokens+512]

        hook_name = f"blocks.{layer}.hook_resid_{hook_type}"
        _, cache = ht.run_with_cache(tokens)
        acts = cache[hook_name][0, :n_tokens].float().cpu()
        del ht, cache
        torch.cuda.empty_cache()
        gc.collect()
        return acts