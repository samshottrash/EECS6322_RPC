import torch

from utils import load_clip, process_cc3m_dataset
from sae import train_autoencoder
import clip

# Just the main file for running the experiments 
# will add as we go

#////////////////////////////////
# Should this part be in main??
# or maybe have a different file to configure everything? Will see as we go.
# will create a different file if the parameters become a lot.

# add comments and explnantions

models = {
    "resnet50": "RN50",
    "vit_b16": "ViT-B/16",
    "vit_l14": "ViT-L/14"
}

clip_model_name = models["resnet50"]
device = "cuda" if torch.cuda.is_available() else "cpu"
#//////////////////////////////

if __name__ == "__main__":
    print("Available CLIP models:", clip.available_models())

    clip_model, preprocess = load_clip(clip_model_name)
    
    print("Processing CC3M dataset...")
    dataset = process_cc3m_dataset(clip_model, preprocess)
    print(f"Finished processing {len(dataset)} images.")

    print("Training sparse autoencoder...")
    neurons_fired = train_autoencoder()
    print(f"Training complete. Neurons fired: {neurons_fired}")

