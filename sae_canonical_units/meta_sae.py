import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_in, hidden, k):
        super().__init__()
        self.enc = nn.Linear(d_in, hidden, bias=True)
        self.b_enc = nn.Parameter(torch.zeros(hidden))
        self.dec = nn.Linear(hidden, d_in, bias=True)
        self.k = k
        
        nn.init.kaiming_uniform_(self.enc.weight)
        self.normalize_decoder()

    def normalize_decoder(self):
        with torch.no_grad():
            self.dec.weight.data = self.dec.weight.data / self.dec.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def forward(self, x):
        pre = self.enc(x) + self.b_enc
        vals, idx = torch.topk(pre, self.k, dim=1)
        acts = torch.zeros_like(pre).scatter_(1, idx, vals)
        acts = F.relu(acts)
        return self.dec(acts), acts

def train_meta_sae(target_sae, hidden=2048, k=4, epochs=1000, lr=3e-4, bs=2048):
    device = target_sae.W_dec.device
    data = target_sae.W_dec.detach().to(torch.float32).to(device)
    d_in = data.shape[1]
    
    sae = TopKSAE(d_in, hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0])
        for i in range(0, data.shape[0], bs):
            b = data[perm[i : i + bs]]
            opt.zero_grad()
            recon, _ = sae(b)
            loss = F.mse_loss(recon, b)
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                recon, _ = sae(data)
                # FIX: Use MSE / Var, not Var(error) / Var(data)
                var_exp = 1 - (recon - data).pow(2).mean() / data.var()
                print(f"meta ep {ep + 1} loss {loss.item():.5f} var_exp {var_exp.item():.3f}")
                
    with torch.no_grad():
        recon, _ = sae(data)
        var_exp = 1 - (recon - data).pow(2).mean() / data.var()
    return sae, var_exp.item()