from absl import flags


flags.DEFINE_integer("image_size", 32, "Image size")

flags.DEFINE_float("learning_rate", 3e-4, "Initial learning rate.")

flags.DEFINE_integer("batch_size", 128, "Batch size for training.")

flags.DEFINE_integer("num_epochs", 100, "Number of epochs to train for.")

flags.DEFINE_integer(
    "checkpoint_epochs", 10, "Number of epochs between checkpoints/summaries."
)

flags.DEFINE_string("dataset_dir", "./datasets", "Directory where dataset is stored.")

flags.DEFINE_integer(
    "num_workers", 8, "Number of concurrent works for PyTorch data loader"
)
