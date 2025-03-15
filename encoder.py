import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import requests
from transformers import CLIPFeatureExtractor
import clip
from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
    sweep,
)

# For the dataset: "We train our sparse autoencoders on the CC3M dataset"
#dataset = load_dataset("conceptual_captions", split="train", streaming=True)

# the paper says they discovered concepts the CLIP model has learnt using an SAE, 
# so we get our pre-trained CLIP model
# "We use CLIP [S16] ResNet-50 [S5], ViT-B/16 [S4], and
# ViT-L/14 [S4] pre-trained feature extractors from the official repository"
model_14, preprocess_14 = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch14")
model_16, preprocess_16 = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16")


# The paper trains the SAE with an L_2 reconstruction loss & an L_1 sparsity 
# regularisation, with a hyperparameter lambda_1
# L1 sparsity coefficient (λ1) {3×10−5, 1.5×10−4, 3×10−4, 1.5×10−3, 3×10−3} for 
# hyperparameter sweeps 

# the paper performs "hyperparameter sweeps
# using a heldout set over the learning rate {1 × 10−5, 5 × 10−5, 1 × 10−4, 5 × 10−4, 1×10−3}"
# SAE with learning rate 5 × 10−4 chosen for CLIP ResNet-50 model. 
sweep_config = SweepConfig(
    parameters=Hyperparameters(
        loss=LossHyperparameters(
            l1_coefficient=Parameter(values=[3e-5, 1.5e-4, 3e-4, 1.5e-3, 3e-3]),
        ),
        optimizer=OptimizerHyperparameters(
            lr=Parameter(values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
        ),
        source_data=SourceDataHyperparameters(
            dataset_path=Parameter("conceptual_captions"),  # CC3M dataset
            context_size=Parameter(256),  # Number of tokens/images to process per batch
            pre_tokenized=Parameter(value=False),  # CC3M is not pre-tokenized
            pre_download=Parameter(value=False)  # Stream instead of downloading
        ),
        autoencoder=AutoencoderHyperparameters(
            expansion_factor=Parameter(values=[2,4,8])
        ),
    ),
    method=Method.RANDOM,
)

