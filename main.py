import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
from modules import BYOL
from modules.transformations import TransformsSimCLR

from absl import app
from flags import flags


FLAGS = flags.FLAGS

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dataset
    train_dataset = datasets.CIFAR10(
        FLAGS.dataset_dir,
        download=True,
        transform=TransformsSimCLR(size=FLAGS.image_size),  # paper 224
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        drop_last=True,
        num_workers=FLAGS.num_workers,
    )

    # model
    resnet = models.resnet50(pretrained=False)
    learner = BYOL(resnet, image_size=FLAGS.image_size, hidden_layer="avgpool")
    learner = learner.to(device)

    # optimizer
    optimizer = torch.optim.Adam(learner.parameters(), lr=FLAGS.learning_rate)

    # solver
    for epoch in range(FLAGS.num_epochs):
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            loss = learner(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()  # update moving average of target encoder

            if step % 100 == 0:
                print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

        if epoch % FLAGS.checkpoint_epochs == 0:
            torch.save(resnet.state_dict(), f"./model-{epoch}.pt")

    # save your improved network
    torch.save(resnet.state_dict(), "./improved-net.pt")


if __name__ == "__main__":
    app.run(main)
