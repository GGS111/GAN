import config
from torch import optim
from utils import load_checkpoint, plot_examples
from model_from_paper import RRDBNet
from model import Generator 



if __name__ == "__main__":
    paper_model = True #Model From Git
    if paper_model:
        gen = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(config.DEVICE)
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        plot_examples(config.PATH_TO_TEST, gen)
    else: 
        gen = Generator(in_channels=3).to(config.DEVICE) #Train my Model
        opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
        load_checkpoint(config.CHECKPOINT_GEN, gen,opt_gen ,config.LEARNING_RATE)
        plot_examples(config.PATH_TO_TEST, gen)