import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_in, hidden, k):
        super().__init__()
        self.enc = nn.Linear(d_in, hidden)
        self.dec = nn.Linear(hidden, d_in)
        self.k = k
        self.dec.weight.data = self.enc.weight.data.T.clone()

    def forward(self, x):
        pre = self.enc(x)
        vals, idx = torch.topk(pre, self.k, dim=1)
        acts = torch.zeros_like(pre).scatter_(1, idx, vals)
        return self.dec(acts), acts

def train_meta_sae(target_sae, hidden=2048, k=4, epochs=200, bs=2048):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = target_sae.W_dec.detach().to(device)
    d_in = data.shape[1]
    sae = TopKSAE(d_in, hidden, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0])
        for i in range(0, data.shape[0], bs):
            b = data[perm[i : i + bs]]
            opt.zero_grad()
            recon, _ = sae(b)
            loss = F.mse_loss(recon, b)
            loss.backward()
            opt.step()
            

    with torch.no_grad():
        recon, _ = sae(data)
        var_exp = 1 - ((data - recon).var() / data.var()).item()
    return sae, var_exp