import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


# TODO: Go through article and add comments explaining

# the autoencoder model, implemented according to details from article
class AutoEncoder(nn.Module):
    def __init__(self, device, acts_dim=1024, exp=8, l1_coeff=3e-4):
        super().__init__()
        dtype = torch.float32
        # torch.manual_seed(42)

        self.acts_dim = acts_dim
        self.latent_dim = acts_dim * exp
        self.l1_coeff = l1_coeff

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(acts_dim, self.latent_dim, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.latent_dim, acts_dim, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(self.latent_dim, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(acts_dim, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        

        self.to(device)

    def encode(self, x):
        x_bias = x - self.b_dec
        linear = x_bias @ self.W_enc + self.b_enc
        z_latents = F.relu(linear)     # shape [B, n_hidden]

        return z_latents

    def decode(self, x):
        recon_x = x @ self.W_dec + self.b_dec
        return recon_x

    def forward(self, x):
        z_latents = self.encode(x)
        recon_x = self.decode(z_latents)

        return recon_x, z_latents
    

    @torch.no_grad()
    def remove_parallel_gradients(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True) + 1e-8
        W_dec_normed = self.W_dec / norms
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

        self.W_dec.data = W_dec_normed


def sae_loss(model, x, x_recon, z_latents):
    recon_loss = (x_recon.float() - x.float()).pow(2).sum(-1).mean(0)
    l1_loss = model.l1_coeff * (z_latents.float().abs().sum())
    total_loss = recon_loss + l1_loss
    return total_loss, recon_loss, l1_loss

# get frequency of neurons for resampling
@torch.no_grad()
def get_neuron_freqs(model, dataloader, device, num_batches=25):
    # z_dim = model.latent_dim

    act_freq_scores = torch.zeros(model.latent_dim, dtype=torch.float32).to(device)
    total = 0

    for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
        
        batch = batch.to(device)        
        _, z_latents = model(batch)
        
        act_freq_scores += (z_latents > 0).sum(0)
        total += z_latents.shape[0]

    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean().item()
    print("Number of dead neurons: ", num_dead)
    return act_freq_scores

# reiniitialize dead neurons
@torch.no_grad()
def re_initialize(indices, model):
    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(model.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(model.W_dec)))
    new_b_enc = (torch.zeros_like(model.b_enc))

    print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)
    model.W_enc.data[:, indices] = new_W_enc[:, indices]
    model.W_dec.data[indices, :] = new_W_dec[indices, :]
    model.b_enc.data[indices] = new_b_enc[indices]
