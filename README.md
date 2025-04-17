# EECS6322_RPC
Reproducibility Challenge for EECS6322 (Neural Networks and Deep Learning) Project

## Below we outline the process to the run the get and get the concepts
 - run to create the conda environment: conda env create --file environment.yaml
 - activate the conda environment with: conda activate eecs6322-rpc

 - The cc3m image dataset clip embeddings have been provided. But to extract them run: python extract_clip_embeds.py
 - To train the sae model on cc3m and discover concepts: python train_sae.py
 - To automatically assign concept names to the learned features of the sae: python auto_concept_naming.py

We visualize the top activating images on the following datasets: cifar10, cifar100 and imagenet.
An example notebook for visualizing the images is provided in: 
 
