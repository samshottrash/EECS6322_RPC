import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


# The autoencoder model, implemented according to the article: https://transformer-circuits.pub/2023/monosemantic-features

class AutoEncoder(nn.Module):

    def __init__(self, device, acts_dim=1024, exp=8, l1_coeff=3e-5):

        """
        Definition for the sparse autoencoder model.
        Weights and biases are explicity defined and accessed

        Args:
            act_dim: the input and out dimenions of the encoders and decoders
            exp: the expansion factor. act_dim x exp gives the expanded dimenion of the latent space
            l1_coeff: the l1 coeffient for the sparsity part of the loss
        """

        super().__init__()
        

        self.acts_dim = acts_dim
        self.latent_dim = acts_dim * exp
        self.l1_coeff = l1_coeff

        # encoder weights and bias
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(acts_dim, self.latent_dim, dtype=torch.float32)))
        self.b_enc = nn.Parameter(torch.zeros(self.latent_dim, dtype=torch.float32))

        # decoder weights and bisa
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.latent_dim, acts_dim, dtype=torch.float32)))
        self.b_dec = nn.Parameter(torch.zeros(acts_dim, dtype=torch.float32))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.to(device)

    def encode(self, x):
        """
        The encoder of the sparse autoencoder
        args:
            x: the clip image embeddings
        """
        x_bias = x - self.b_dec
        linear = x_bias @ self.W_enc + self.b_enc
        z_latents = F.relu(linear)     # shape [B, latent_dim] = [, 8192] for expansion factor 8

        return z_latents

    def decode(self, x):
        """
        Decoder for the SAE
        Args:
            x: the latent space computed from the encoder

        returns:
            recon_x: the reconstructed clip embeddings
        """

        recon_x = x @ self.W_dec + self.b_dec
        return recon_x

    def forward(self, x):
        """
        forward call of the SAE. Runs the model encoder and decoder.

        Args:
            x: input, the clip embeddings

        Returns: 
            recon_x: the reconstrcucted clip embdeddings
            z_latents: the latent (hidden layer activations) space of the sae. The hidden layer activations are our learned features.
        """

        z_latents = self.encode(x)
        recon_x = self.decode(z_latents)

        return recon_x, z_latents
    

    @torch.no_grad()
    def remove_parallel_gradients(self):
        """
        remove any gradient information parallel to the dictionary vectors before applying the gradient step
        """
        norms = self.W_dec.norm(dim=-1, keepdim=True) + 1e-8
        W_dec_normed = self.W_dec / norms
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

        # Set decoder weights to norm again after the gradient update
        self.W_dec.data = W_dec_normed


def sae_loss(model, x, x_recon, z_latents):
    """
    MSE loss plus an L1 penalty to encourage sparsity.

    args:
        model: the sae model
        x: the input clip embedding
        x_recon: the reconstructed clip embedding, computed by the model
        z_latent: the latent dimenions of the sae model

    returns: 
        total loss, recon_loss, l1_loss

    """

    recon_loss = (x_recon.float() - x.float()).pow(2).sum(-1).mean(0)
    l1_loss = model.l1_coeff * (z_latents.float().abs().sum())
    total_loss = recon_loss + l1_loss

    return total_loss, recon_loss, l1_loss

# get frequency of neurons for resampling: to help find more features and achieve a lower total loss
@torch.no_grad()
def get_neuron_freqs(model, dataloader, device, num_batches=25):

    """ 
    Periodically check for neurons which have not fired in a significant number of steps 
    and reset the encoder weights (in re_initialize() method) on the dead neurons
    """

    # z_dim = model.latent_dim

    act_freq_scores = torch.zeros(model.latent_dim, dtype=torch.float32).to(device)
    total = 0

    # for a num of batches, get the number of times a neuron has fired
    for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
        
        batch = batch.to(device)        
        _, z_latents = model(batch)
        
        act_freq_scores += (z_latents > 0).sum(0)
        total += z_latents.shape[0]

    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean().item()
    # print("Number of dead neurons: ", num_dead)
    return act_freq_scores


@torch.no_grad()
def re_initialize(indices, model):
    """
    reiniitialize dead neurons

    Args:
        indices: indices of those neurons with zero activation frequency
        model: the Sae model
    """

    new_W_enc = (torch.nn.init.kaiming_uniform_(torch.zeros_like(model.W_enc)))
    new_W_dec = (torch.nn.init.kaiming_uniform_(torch.zeros_like(model.W_dec)))
    new_b_enc = (torch.zeros_like(model.b_enc))
    # print(new_W_dec.shape, new_W_enc.shape, new_b_enc.shape)

    # randomize the weights of dead neurons so they can learn.
    model.W_enc.data[:, indices] = new_W_enc[:, indices]
    model.W_dec.data[indices, :] = new_W_dec[indices, :]
    model.b_enc.data[indices] = new_b_enc[indices]
