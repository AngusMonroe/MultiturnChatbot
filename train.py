# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse
import numpy as np
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from utils.util import *
from model import *
from eval import *
from torchtext.vocab import GloVe, Vectors
from torchtext import data, datasets, vocab

optparser = optparse.OptionParser()
optparser.add_option(
    "-C", "--corpus_name", default="cornell-movie-dialogs-corpus",
    help="Corpus name"
)
optparser.add_option(
    "-A", "--attn_model", choices=['dot', 'general', 'concat'], default="general",
    help="Attention model"
)
optparser.add_option(
    "-H", "--hidden_size", default=300,
    help="Dimensionality of the glove word vector"
)
optparser.add_option(
    "--encoder_n_layers", default=2,
    help="Layer number of encoder"
)
optparser.add_option(
    "--decoder_n_layers", default=2,
    help="Layer number of decoder"
)
optparser.add_option(
    "-D", "--dropout", default=0.1,
    help="Dropout"
)
optparser.add_option(
    "-B", "--batch_size", default=256,
    help="Batch size"
)
optparser.add_option(
    "-C", "--clip", default=50.0,
    help="Clip gradients: gradients are modified in place"
)
optparser.add_option(
    "--teacher_forcing_ratio", default=1.0,
    help="Teacher forcing ratio"
)
optparser.add_option(
    "--decoder_learning_ratio", default=5.0,
    help="Decoder learning ratio"
)
optparser.add_option(
    "-L", "--learning_rate", default=0.0001,
    help="Learning rate"
)
optparser.add_option(
    "-N", "--n_iteration", default=4000,
    help="Epoch number"
)
optparser.add_option(
    "-P", "--print_every", default=10,
    help="Print frequency"
)
optparser.add_option(
    "-S", "--save_every", default=1000,
    help="Save frequency"
)
optparser.add_option(
    "-G", "--glove_path", default='model/glove/glove.6B.300d.txt',
    help="Path of Glove word vector file"
)


opts = optparser.parse_args()[0]

corpus_name = opts.corpus_name
# Configure models
# model_name = 'cb_model'
attn_model = opts.attn_model
hidden_size = opts.hidden_size
encoder_n_layers = opts.encoder_n_layers
decoder_n_layers = opts.decoder_n_layers
dropout = opts.dropout
batch_size = opts.batch_size
# Configure training/optimization
clip = opts.clip
teacher_forcing_ratio = opts.teacher_forcing_ratio
decoder_learning_ratio = opts.decoder_learning_ratio
learning_rate = opts.learning_rate
n_iteration = opts.n_iteration
print_every = opts.print_every
save_every = opts.save_every
glove_path = opts.glove_path

corpus = os.path.join("data", corpus_name)
# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")
trainfile = os.path.join(corpus, "train.txt")

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 10

# Load/Assemble voc and pairs
cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)

# TEXT.build_vocab(pos, vectors=GloVe(name='6B', dim=300))
save_dir = os.path.join("model", corpus_name)
voc, pairs = loadPrepareData(corpus, corpus_name, trainfile, datafile, save_dir, MAX_LENGTH)
# Print some pairs to validate
# print("\npairs:")
# for pair in pairs[:10]:
#     print(pair)

# Trim voc and pairs
pairs = trimRareWords(voc, pairs)

# Example for validation
# small_batch_size = 5
# batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
# input_variable, lengths, target_variable, mask, max_target_len = batches
#
# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t], device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, loadFilename):

    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            save_path = os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint'))
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, save_path)
    return save_path


# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
weight_matrix = Vectors(glove_path)
# weight_matrix = vocab.GloVe('6B', 300)
voc.getEmb(weight_matrix)
# print(torch.FloatTensor(np.array(voc.index2emb)).size())
embedding.weight.data.copy_(torch.FloatTensor(np.array(voc.index2emb)))
embedding.weight.requires_grad = False
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate)
decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
save_path = trainIters(voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, loadFilename)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

test_path = os.path.join(corpus, "test.txt")
res_path = re.sub(r'\.tar', '.txt', save_path)
evaluateFile(encoder, decoder, searcher, voc, test_path, res_path, max_length=MAX_LENGTH)
