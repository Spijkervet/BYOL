# BYOL - Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning
PyTorch implementation of "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" by J.B. Grill et al.

[Link to paper](https://arxiv.org/abs/2006.07733)

This repository includes a practical implementation of BYOL with:
- **Distributed Data Parallel training**
- Benchmarks on vision datasets (CIFAR-10 / STL-10)
- Support for PyTorch **<= 1.5.0**

Open BYOL in Google Colab Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B68Ag_oRB0-rbb9AwC20onmknxyYho4B?usp=sharing)

## Results
These are the top-1 accuracy of linear classifiers trained on the (frozen) representations learned by BYOL:

| Method  | Batch size | Image size | ResNet | Projection output dim. | Pre-training epochs | Optimizer | STL-10 | CIFAR-10
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| BYOL + linear eval.  | 192 | 224x224 | ResNet18 | 256 | 100 | Adam | _ | **0.832** | 
| Logistic Regression | - | - | - | - | - | - | 0.358 | 0.389 |


## Installation
```
git clone https://github.com/spijkervet/byol --recurse-submodules -j8
pip3 install -r requirements.txt
python3 main.py
```


## Usage
### Using a pre-trained model
The following commands will train a logistic regression model on a pre-trained ResNet18, yielding a top-1 accuracy of 83.2% on CIFAR-10.
```
curl https://github.com/Spijkervet/BYOL/releases/download/1.0/resnet18-CIFAR10-final.pt -L -O
rm features.p
python3 logistic_regression.py --model_path resnet18-CIFAR10-final.pt
```

### Pre-training
To run pre-training using BYOL with the default arguments (1 node, 1 GPU), use:
```
python3 main.py
```

Which is equivalent to:
```
python3 main.py --nodes 1 --gpus 1
```
The pre-trained models are saved every *n* epochs in \*.pt files, the final model being `model-final.pt`

### Finetuning
Finetuning a model ('linear evaluation') on top of the pre-trained, frozen ResNet model can be done using:
```
python3 logistic_regression.py --model_path=./model_final.pt
```

With `model_final.pt` being file containing the pre-trained network from the pre-training stage.

## Multi-GPU / Multi-node training
Use `python3 main.py --gpus 2` to train e.g. on 2 GPU's, and `python3 main.py --gpus 2 --nodes 2` to train with 2 GPU's using 2 nodes.
See https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html for an excellent explanation.

## Arguments
```
--image_size, default=224, "Image size"
--learning_rate, default=3e-4, "Initial learning rate."
--batch_size, default=42, "Batch size for training."
--num_epochs, default=100, "Number of epochs to train for."
--checkpoint_epochs, default=10, "Number of epochs between checkpoints/summaries."
--dataset_dir, default="./datasets", "Directory where dataset is stored.",
--num_workers, default=8, "Number of data loading workers (caution with nodes!)"
--nodes, default=1, "Number of nodes"
--gpus, default=1, "number of gpus per node"
--nr, default=0, "ranking within the nodes"
```
