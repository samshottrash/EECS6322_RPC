import torch
import numpy as np
from PIL import Image
import requests
from transformers import CLIPVisionModel, CLIPImageProcessor

# For the dataset: "We train our sparse autoencoders on the CC3M dataset"

# the paper says they discovered concepts the CLIP model has learnt using an SAE, 
# so we get our pre-trained CLIP model
# "We use CLIP [S16] ResNet-50 [S5], ViT-B/16 [S4], and
# ViT-L/14 [S4] pre-trained feature extractors from the official repository"
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

# CLIP processor has both the feature extractor and the tokenizer
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# the sparse autoencoder is formed like the ones here: 
# https://transformer-circuits.pub/2023/monosemantic-features#setup-autoencoder

class SAE(torch.nn.Module):
    def __init__(self, d, h):
        '''
        from the article:
        Our sparse autoencoder is a model with a bias at the input, 
        a linear layer with bias and ReLU for the encoder, and then another 
        linear layer and bias for the decoder.
        '''

        super(SAE, self).__init__()

        # the hidden rep of f(a), where f is the encoder and a is input is larger than 
        # CLIP embedding space. d in article is CLIP's embedding space, h is hidden 
        # representation
        # input a for the SAE is the output features from CLIP model's vision encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d, h),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(h, d)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# need to figure out d for now
d = 0    
h = 256

sparse_autoencoder = SAE(d, h)

# The paper trains the SAE with an L_2 reconstruction loss & an L_1 sparsity 
# regularisation, with a hyperparameter lambda_1
# L1 sparsity coefficient (λ1) {3×10−5, 1.5×10−4, 3×10−4, 1.5×10−3, 3×10−3} for 
# hyperparameter sweeps 
lambda_1 = 1e-3
criterion = torch.nn.MSELoss(reduction=sum)
sparsity_loss = lambda_1 * torch.norm(sparse_autoencoder.encoder[0].weight, p=1)  

# set up optimizer, the paper performs "hyperparameter sweeps
# using a heldout set over the learning rate {1 × 10−5, 5 × 10−5, 1 × 10−4, 5 × 10−4, 1×10−3}"
# SAE with learning rate 5 × 10−4 chosen for CLIP ResNet-50 model
lr = 5e-4

