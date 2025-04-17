# EECS6322_RPC
Reproducibility Challenge for EECS6322 (Neural Networks and Deep Learning) Project

In this repo, we attempt to reproduce two main contributions described in the paper: 

[**Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery**](https://arxiv.org/abs/2407.14499)

The two contributions are: 
  1)  Use sparse autoencoders (SAEs) to discover concepts learnt by CLIP
  2)  Automatically name the discovered concepts

## Below we outline the process to run the files and get the concepts

 - To train the sae model on the cc3m clip embeddings and discover concepts: `python train_sae.py`
 - To automatically assign concept names to the learned features of the sae: `python auto_concept_naming.py`

We visualize the top activating images on the following datasets: 
- [cifar10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) 
- [cifar100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
- [ImageNet](https://www.image-net.org/)

An example notebook for visualizing the images is provided in: [notebook](visualizing.ipynb)
 
