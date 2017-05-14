from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import click

from model import model

@click.command()
@click.option('--data_path', default='data/input.txt', type=str, help="Path to file with text data.")
@click.option('--save_dir', default='save/', type=str, help='Directory to store checkpointed models')
@click.option('--checkpoint_file', default='model.ckpt', type=str, help='File to store checkpointed models')
@click.option('--num_epochs', default=10, type=int, help="Number of epochs.")
@click.option('--training_epochs', default=100000, type=int, help="Number of epochs.")
@click.option('--num_layers', default=1, type=int, help="Number of layers in encoder.")
@click.option('--batch_size', default=128, type=int, help="The size of batch.")
@click.option('--num_steps', default=10, type=int, help="The size time steps.")
@click.option('--max_len', default=100, type=int, help='Maximum sequence length in encoder and decoder.'
                                                       'Lines with higher length will be cutted.')
@click.option('--num_hidden', default=128, type=int, help="Hidden size of the cell.")
@click.option('--restore', default=True, type=bool, help="Restore previous saved model.")
@click.option('--train', default=True, type=bool, help="Training or tesing model.")
@click.option('--embedding_size', default=128, type=int, help="The size of word embeddings.")
@click.option('--n_components_encoder', default=128, type=int, help="Hidden size of the cell.")
@click.option('--n_components_decoder', default=128, type=int, help="Hidden size of the cell.")
@click.option('--num_latent_hidden', default=5, type=int, help="Hidden size of the cell.")
@click.option('--vocabulary_size', default=48934, type=int, help="Size of vocabulary. Most frequent words are used.")
@click.option('--num_samples', default=512, type=int, help="Number of samples in sampled softmax.")
@click.option('--learning_rate', default=0.0001, type=float, help="Initial learning rate.")
@click.option('--decay_rate', default=0.99, type=float, help="Exponential decay rate.")
@click.option('--grad_clip', default=5.0, type=float, help="Value for gradient clipping.")
@click.option('--save_every', default=100, type=int, help="Number of batch steps before creating a model checkpoint")

def main(**kwargs):
	args = Struct(**kwargs)
	vae_lstm = model.VAELSTM(args)

if __name__ == "__main__":
	class Struct:
	    def __init__(self, **entries):
	        self.__dict__.update(entries)
	main()