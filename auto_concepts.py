# For now just base code
# need to modify for our purposes.
# 

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from sae import AutoEncoder
from datasets import Dataset

# Added by me
import os
from torchtext.data import get_tokenizer

# Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# prepare vocab
# cwith open('20k_vocab.txt', 'r') as f:
#    vocab = [line.strip() for line in f if line.strip()]

# print(vocab)

# load decoder weight from trained SAE
# from the paper: "each of the SAE neurons c is assigned a specific dictionary vector pc, corresponding to a column of the decoder weight matrix"
sae = AutoEncoder().eval()
decoder_weights = sae.W_dec.data.T

""" def get_text_embedding(word):
    tokens = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**tokens).last_hidden_state[:, 0, :]  # CLS token
    return embedding

text_embeddings = torch.cat([get_text_embedding(w) for w in vocab], dim=0)  # (V, d) """

text_embeddings = Dataset.from_file("./text_embeddings/data-00000-of-00001.arrow")


##################
# Compute cosine similarity between image embedding and language embedding
cosine = nn.CosineSimilarity(dim=1)

# the decoder weights are columns while the embeddings are rows
decoder_exp = decoder_weights.unsqueeze(1)       # (hidden_dim, 1, d)
text_emb_exp = text_embeddings.unsqueeze(0)      # (1, vocab_size, d)

# Step 2: Compute cosine similarity for all pairs
# Output shape: (hidden_dim, vocab_size)
similarities = cosine(decoder_exp.expand(-1, text_emb_exp.size(1), -1),
                   text_emb_exp.expand(decoder_exp.size(0), -1, -1))

# Step 3: Get the best matching word index for each neuron
closest_indices = torch.argmax(similarities, dim=1)

# Step 4: Map to vocab
concept_names = [vocab[i] for i in closest_indices]

""" cos_values = {}
for text_idx, text in enumerate(text_embeddings):
    similar_imgs = {}
    for weight_idx, weight in enumerate(decoder_weights):
        # print(text.shape)
        # print(img.shape)
        cos_val = cosine(decoder_exp, text_emb_exp)
        similar_imgs[img_idx] = cos_val.item()

    cos_values[text_idx] = similar_imgs


#  5) The image with the highest cosine similarity is the image you matched
# print index of images with highest cosine similarity

# sort the cosine values in descending order
top_image = {}
for text_idx, cos_sims in cos_values.items():
    sorted_cosines = sorted(cos_sims.items(), key=lambda x: x[1], reverse=True)

    # get the index of the top image with the highest cosine value
    top_image[text_idx] = sorted_cosines[0][0] """

# print(top_image)

# Your code should save out the images that you find
""" save_dir = "Part1_images"
os.makedirs(save_dir, exist_ok=True)

for text_idx, img_idx in top_image.items():
    found_img = ds[img_idx]
    best_img = torchvision.transforms.ToPILImage()(found_img) 
    img_path = os.path.join(save_dir, f"text_{text_idx}_best_image_is_{img_idx}.png")
    best_img.save(img_path)

    # AND
    # You should print out what idx each image is found at
    print(f"For Text description {text_idx}: Image found at image index {img_idx}") """