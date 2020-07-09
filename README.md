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

### Arguments
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