# BYOL
PyTorch implementation of "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" by J.B. Grill et al.
Added support for PyTorch <= 1.5.0 and practical dataset implementation (CIFAR-10).

## Installation
```
git clone https://github.com/spijkervet/byol --recurse-submodules -j8
pip3 install -r requirements.txt
python3 main.py
```


## Usage
Define your training parameters in `flags.py`.
Use `python3 main.py` to run pre-training using BYOL with the default arguments (1 node, 1 GPU)

## Multi-GPU / Multi-node training
Use `python3 main.py --gpus 2` to train e.g. on 2 GPU's, and `python3 main.py --gpus 2 --nodes 2` to train with 2 GPU's using 2 nodes.
See https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html for an excellent explanation.