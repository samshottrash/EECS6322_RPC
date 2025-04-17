import torch
from utils import load_clip, extract_features, get_text_embeds

"""
script tp extract the image clip embeddings from the cc3m dataset

"""
if __name__ == "__main__":

    models = {
        "resnet50": "RN50",
        "vit_b16": "ViT-B/16",
        "vit_l14": "ViT-L/14"
    }

    clip_model_name = models["resnet50"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device: ", device)

    # load clip model
    clip_model, preprocess = load_clip(clip_model_name, device)

    # //////////////////////////////
    # extract the image embeddings using clip (from the utils file)
    # path to the CSV file containing the local image paths.
    csv_file = "cc3m_local.csv"  

    print("Extracting image features from the cc3m dataset...")
    processed_dataset = extract_features(csv_file, clip_model, preprocess, device)
    # print("Image feature extraction complete.")
    print(f"Processed dataset contains {len(processed_dataset)} samples.")

    # //////////////////////////////
    # extract the clip text embeddings from the vocab
    # ONLY NEEDED TO RUN ONCE. If need to extract again, then just uncomment below

    """    
    # vocab file containing 20k words
    text_file = "20k_vocab.txt"

    print("Extracting text embeddings.")
    text_embeddings = get_text_embeds(clip_model, text_file, device)
    print("Text embeddings shape:", text_embeddings.shape)
    
    """