import config
from torch import optim
from utils import load_checkpoint, plot_examples
from model_from_paper import Generator

if __name__ == "__main__":
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    load_checkpoint(
        config.CHECKPOINT_GEN,
        gen,
        opt_gen,
        config.LEARNING_RATE,
    )
    plot_examples(config.PATH_TO_TEST, gen)