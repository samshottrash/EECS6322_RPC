import torch
import numpy as np
from PIL import Image
import requests
from transformers import CLIPVisionModel, CLIPImageProcessor

# the paper says they discovered concepts the CLIP model has learnt using an SAE, 
# so we get our pre-trained CLIP model
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

# CLIP processor has both the feature extractor and the tokenizer
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# the sparse autoencoder is formed like the ones here: 
# https://transformer-circuits.pub/2023/monosemantic-features#setup-autoencoder

class SAE(torch.nn.Module):
    def __init__(self):
        '''
        from the article:
        Our sparse autoencoder is a model with a bias at the input, 
        a linear layer with bias and ReLU for the encoder, and then another 
        linear layer and bias for the decoder.
        '''

        super(SAE, self).__init__()

        # the hidden rep of f(a), where f is the encoder and a is input is larger than 
        # CLIP embedding space
        # input a for the SAE is the output features from CLIP model's vision encoder
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(),
            torch.nn.ReLU(),
            torch.nn.Linear()
        )