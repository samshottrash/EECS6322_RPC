import clip
import torch
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset

# from config import device

'''
MAKE SURE TO EDIT!!! COMPARE TO OLD CLIP_UTILS FILE

This file handles things like loading clip and processing the dataset
Need to check and make sure it si in sync with the notebook. Add comments from notebook

TO DO:
- load vocab
- get text embeddings using clip
- implement automatic concept naming using cosine similarity: maybe in a different file
- visualize results: also maybe different file
- add metric watching like tensorboard and wandb?
'''

def load_clip(model_name, device):
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

# get the dataset file that saves the path for each image
def get_dataset(csv_file, preprocess, batch_size=32):
    ds = load_dataset("csv", data_files=csv_file)["train"]

    # Use partial to pass our preprocess function into the mapping function.
    # ds = ds.map(partial(load_and_preprocess, preprocess=preprocess),
    #             batched=True, batch_size=batch_size)
    
    ds = ds.map(lambda x: load_and_preprocess(x, preprocess))

    
    # Explode the batched columns so that each image becomes an individual example.
    # ds = ds.explode(["pixel_values", "image_path"])
    # Filter out examples where image failed to load (None).
    ds = ds.filter(lambda x: x["pixel_values"] is not None)
    # Set format so that "pixel_values" becomes a PyTorch tensor.
    ds.set_format("torch", columns=["pixel_values", "image_path"])

    return ds

# Define image loading + preprocessing inside map
def load_and_preprocess(image, preprocess):
    # pixel_values = []
    #for path in batch["image_path"]:
    try:
        img = Image.open(image["image_path"]).convert("RGB")
        processed = preprocess(img)

    except Exception as e:
        # In case of error, create a tensor of zeros (CLIP expects [3, 224, 224])
        processed = None
        # print(f"Error processing {image["image_path"]}: {e}")
    # pixel_values.append(processed)
    image["pixel_values"] = processed
    # return {"pixel_values": pixel_values, "image_path": image["image_path"]}
    return image


def extract_features(csv_file, clip_model, preprocess, device):
    dataset = get_dataset(csv_file, preprocess, batch_size=32)
    # print("Dataset examples after exploding and filtering:", dataset[:5])

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_features = []
    all_image_paths = []

    clip_model.eval()
    for batch in tqdm(dataloader, desc="Extracting CLIP features"):

        images = batch["pixel_values"]
        paths = batch["image_path"]
        images = images.to(device)
        with torch.no_grad():
            features = clip_model.encode_image(images)  # # shape: [B, D]
        for feature, path in zip(features, paths):
            all_features.append(feature.cpu().numpy())
            all_image_paths.append(path)

    processed_dataset = Dataset.from_dict({
        "image_path": all_image_paths,
        "clip_features": all_features
    })

    # just trying different things and formats
    processed_dataset.save_to_disk("cc3m_clip_features") #hugging face
    torch.save(processed_dataset, "image_embeds.pt")  # pytorch format
    np.save("image_embeds.npy", processed_dataset)    # numpy format

    print("Clip image embeddings have been saved to directory")

    return processed_dataset
    
# MODIFY - compare to clip_utils for correct format
def get_text_embeds(clip_model, file_path, device):
    # read the words from the text file
    batch_size= 64
    with open(file_path, 'r', encoding='utf-8') as file:
        texts = file.readlines()

    all_embeddings = []
    # Process texts in batches with a progress bar.
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting text embeddings"):
        batch_texts = texts[i:i+batch_size]
        # Tokenize the texts (returns a tensor).
        text_tokens = clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            batch_embeddings = clip_model.encode_text(text_tokens)
        all_embeddings.append(batch_embeddings.cpu())
    
    # Concatenate all batch embeddings into one tensor.
    text_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Save as a Hugging Face Dataset (converted from the tensor to a list).
    # Note: The Hugging Face dataset format saves to a folder.
    hf_dataset = Dataset.from_dict({"text_embedding": text_embeddings.tolist()})
    hf_dataset.save_to_disk("text_embeddings")
    
    # Save in PyTorch format.
    torch.save(text_embeddings, "clip_text_embeds.pt")
    
    # Save in NumPy format.
    np.save("clip_text_embeds.npy", text_embeddings.numpy())

    print("Clip text embeddings have been saved to directory")
    
    return text_embeddings