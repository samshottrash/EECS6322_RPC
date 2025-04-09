import clip
import torch
from PIL import Image
from io import BytesIO
import requests
from datasets import load_dataset
from config import device

'''
This file handles things like loading clip and processing the dataset
Need to check and make sure it si in sync with the notebook. Add comments from notebook

TO DO:
- load vocab
- get text embeddings using clip
- implement automatic concept naming using cosine similarity: maybe in a different file
- visualize results: also maybe different file
'''

def load_clip(model_name):
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

def extract_features(clip_model, preprocess, image):
    image_url = image["image_url"]

    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)

        return {"clip_features": image_features}

    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return {"clip_features": None}

def process_cc3m_dataset(clip_model, preprocess):
    dataset = load_dataset("conceptual_captions", split="train")

    def feature_mapper(example):
        return extract_features(clip_model, preprocess, example)

    # or dataset = dataset.map(extract_features(clip_model, preprocess))
    dataset = dataset.map(feature_mapper)
    dataset = dataset.filter(lambda x: x["clip_features"] is not None)
    dataset.save_to_disk("cc3m_clip_features")
    return dataset

def load_text_vocab():

    return

def get_text_embeds():

    return
