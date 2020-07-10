import os
import argparse
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

from modules.transformations import TransformsSimCLR
from process_features import get_features, create_data_loaders_from_arrays

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=3e-3, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=768, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=300, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resnet_version", default="resnet18", type=str, help="ResNet version."
    )
    parser.add_argument(
        "--checkpoint_epochs",
        default=10,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers (caution with nodes!)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # data loaders
    train_dataset = datasets.CIFAR10(
        args.dataset_dir,
        download=True,
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )

    test_dataset = datasets.CIFAR10(
        args.dataset_dir,
        train=False,
        download=True,
        transform=TransformsSimCLR(size=args.image_size).test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # pre-trained model
    if args.resnet_version == "resnet18":
        resnet = models.resnet18(pretrained=False)
    elif args.resnet_version == "resnet50":
        resnet = models.resnet50(pretrained=False)
    else:
        raise NotImplementedError("ResNet not implemented")


    # resnet.load_state_dict(torch.load(args.model_path, map_location=device))
    resnet = resnet.to(device)

    num_features = list(resnet.children())[-1].in_features

    # throw away fc layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    n_classes = 10 # CIFAR-10 has 10 classes

    # fine-tune model
    logreg = nn.Sequential(nn.Linear(num_features, n_classes))
    logreg = logreg.to(device)

    # loss / optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

    # compute features (only needs to be done once, since it does not backprop during fine-tuning)
    if not os.path.exists("features.p"):
        print("### Creating features from pre-trained model ###")
        (train_X, train_y, test_X, test_y) = get_features(
            resnet, train_loader, test_loader, device
        )
        pickle.dump(
            (train_X, train_y, test_X, test_y), open("features.p", "wb"), protocol=4
        )
    else:
        print("### Loading features ###")
        (train_X, train_y, test_X, test_y) = pickle.load(open("features.p", "rb"))


    train_loader, test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, 2048 
    )

    # Train fine-tuned model
    for epoch in range(args.num_epochs):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_loader):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))


    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))




