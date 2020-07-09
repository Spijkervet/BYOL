# BYOL
PyTorch implementation of "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" by J.B. Grill et al.
Added support for PyTorch <= 1.5.0 and practical dataset implementation (CIFAR-10).

## Installation
```
git clone https://github.com/spijkervet/byol --recurse-submodules -j8
pip3 install -r requirements.txt
```


## Usage
Define your training parameters in `flags.py`.
Use `python3 main.py` to run pre-training using BYOL.