
# Code to train the SAE on the cc3m clip embeddings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_from_disk

# from utils import embedDataset
from sae import AutoEncoder, sae_loss, get_neuron_freqs, re_initialize
from torch.utils.tensorboard import SummaryWriter


# Function to train the SAE
def train_autoencoder(model, dataloader, device, num_epochs = 200, lr=5e-4):

    writer = SummaryWriter(log_dir="./runs/sae_experiment")
    save_dir = Path("./sae_checkpoints")
    save_dir.mkdir(exist_ok=True)

    global_step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.99))
    # act_freq_scores_list = []

    # start training
    for epoch in range(num_epochs):
        model.train()
        for x in tqdm(dataloader):
            x = x.to(device)

            recon_x, z_latents = model(x)
            loss, recon_loss, l1_loss= sae_loss(model, x, recon_x, z_latents)
            loss.backward()

            model.remove_parallel_gradients()
            optimizer.step()
            optimizer.zero_grad()

            # dont forget to make logs using tensorboard
            writer.add_scalar("Loss/Total", loss, global_step)
            writer.add_scalar("Loss/Reconstruction_loss", recon_loss, global_step)
            writer.add_scalar("Loss/L1_loss(Sparsity)", l1_loss, global_step)

            global_step += 1

        if epoch % 10 == 0:
            # resampling for dead neurons every 10 epochs
            freqs = get_neuron_freqs(model, dataloader, device)
            to_be_reset = (freqs<10**(-5.5))
            # print("Resetting neurons!", to_be_reset.sum())
            re_initialize(to_be_reset, model)

            # log number of dead nuerons to tensorboard
            dead_amount = (freqs==0).float().mean().item()
            writer.add_scalar("Dead_neurons", dead_amount, epoch)

            # save model checkpoint (every 10 epochs) for future loading
            path = save_dir / f"sae_epoch_{epoch}.pth"
            torch.save(model.state_dict(), path)

    print("Done training")

    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ",device)

    # Load the cc3m clip embeddings dataset from disk.
    embeddings = load_from_disk("cc3m_clip_features")
    print(f"Loaded cc3m clip embeddings with {len(embeddings)} examples.")

    # Extract the clip_features np.array
    dataset = torch.tensor(np.array(embeddings["clip_features"]), dtype=torch.float32)
    # print("Embeddings array shape:", dataset.shape)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # train the autoencoder using the function defined above
    sae_model = AutoEncoder(device)
    SAE_trained_model = train_autoencoder(sae_model, dataloader, device)

    # save the final returned model
    save_dir = Path("./sae_checkpoints")
    save_dir.mkdir(exist_ok=True)
    final_save_path = save_dir / "sae_final.pth"
    torch.save(SAE_trained_model.state_dict(), final_save_path)