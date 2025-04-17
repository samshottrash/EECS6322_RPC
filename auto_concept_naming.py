import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from sae import AutoEncoder
from datasets import Dataset, load_from_disk
import torch.nn.functional as F
import pandas as pd

"""
Script for the automatic naming of concepts discovered by the sparse autoencoder

"""
# cosine similarity functionn
# to calculate the cosine similarity between the text embeddings and sae decoder weights
def cosine(weights, texts):
    weights_norm = F.normalize(weights, p=2, dim=1)
    texts_norm = F.normalize(texts, p=2, dim=1)
    similarities = torch.mm(weights_norm, texts_norm.T)
    return similarities


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    # load the precomputed clip text embeddings
    vocab_dataset = load_from_disk("text_embeddings")
    print(f"Loaded text embeddings with {len(vocab_dataset)} examples.")

    text_embeds = torch.tensor(np.array(vocab_dataset["text_embedding"]), dtype=torch.float32).to(device)
    # print(text_embeds.shape)

    # load the autoenocder model from saved checkpoint
    sae = AutoEncoder(device)
    checkpoint = torch.load("sae_checkpoints/sae_final.pth", map_location=device)

    sae.load_state_dict(checkpoint)
    sae.eval()

    # load decoder weights from trained SAE
    # from the paper: "each of the SAE neurons c is assigned a specific dictionary vector pc, corresponding to a column of the decoder weight matrix"
    decoder_weights = sae.W_dec.data
    # print(decoder_weights.shape)

    # get the similarities between the texts and weights
    similarities = cosine(decoder_weights, text_embeds)
    # print(similarities.shape)

    # Get the index of the best matching word for each neuron in the sae latent space
    closest_indices = torch.argmax(similarities, dim=1)
    # closest_indices.shape

    # prepare vocab: the words to be assigned
    with open('20k_vocab.txt', 'r') as f:
        vocab = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(vocab)} vocabulary words.")
    # print(vocab[5684])

    # save assigned name to an external csv file
    neurons = list(range(decoder_weights.shape[0]))   # 8192 neurons
    concepts = [vocab[i] for i in closest_indices.tolist()]

    concept_df = pd.DataFrame({"neuron_index": neurons, "assigned_concepts": concepts})
    save_path = "assigned_concept_names.csv"
    concept_df.to_csv(save_path, index=False)
    print(f"Saved learned concept names for each neuron to {save_path}")