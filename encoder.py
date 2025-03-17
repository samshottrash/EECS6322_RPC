import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import requests
import clip
from datasets import load_dataset
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

print(clip.available_models())
models = {
    "resnet50": "RN50",
    "vit_b16": "ViT-B/16",
    "vit_l14": "ViT-L/14"
}
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip(model_name):

    model, preprocess = clip.load(model_name, device=device)
    
    return model, preprocess

clip_model_name = models["resnet50"]
clip_model, preprocess = load_clip(clip_model_name)

# use model.encode_image() to get the image features of each img in CC3M
def get_features(image):

    # prepare image for feature extraction
    print("here")
    print(image)
    image_input = preprocess(Image.open(image["image_url"])).unsqueeze(0).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
    
    return {"clip_features": image_features}

dataset = load_dataset("conceptual_captions", split="train")
dataset = dataset.map(get_features)
dataset.save_to_disk("cc3m_clip_features")  # Save processed dataset

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
        source_model=SourceModelHyperparameters(
            name=Parameter("openai/clip"), # idk if i should specify the model
            cache_names=Parameter(["vision_model.encoder.layers.11"]),  # Extract from last layer
            hook_dimension=Parameter(768 if clip_model_name == "ViT-B/16" else 1024)
        ),
        source_data=SourceDataHyperparameters(
            dataset_path=Parameter("cc3m_clip_features"),  # CC3M dataset
            context_size=Parameter(256),  # Number of tokens/images to process per batch
            pre_tokenized=Parameter(value=False),  # CC3M is not pre-tokenized
            pre_download=Parameter(value=False),  # Stream instead of downloading
            tokenizer_name=Parameter("openai/clip-vit-base-patch32")
        ),
        autoencoder=AutoencoderHyperparameters(
            expansion_factor=Parameter(values=[2,4,8])
        ),
        num_epochs = Parameter(200),
        resample_interval = Parameter(10)
    ),
    method=Method.RANDOM,
)

