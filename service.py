# -*- coding:utf-8 -*-
import os
import optparse
import torch
from utils.util import *
from model import *
from eval import *

MAX_LENGTH = 10

optparser = optparse.OptionParser()
optparser.add_option(
    "--model", default='',
    help="Path of model file"
)

opts = optparser.parse_args()[0]

# Configure model
model = opts.model


def init_model(model_path):
    if not model_path:
        raise RuntimeError('ModelNotFoundError')
    seq2seq = torch.load(model_path)
    # Use appropriate device
    seq2seq.to(device)
    print('Models built and ready to go!')

    # Set dropout layers to eval mode
    seq2seq.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(seq2seq)
    return seq2seq, searcher


def init_voc(corpus, corpus_name, datafile, trainfile, save_dir):
    voc, pairs = loadPrepareData(corpus, corpus_name, trainfile, datafile, save_dir, MAX_LENGTH)
    return voc


def get_para_from_seq2seq(seq2seq):
    corpus_name = seq2seq.opts.corpus_name
    data_file = seq2seq.opts.data_file
    train_file = seq2seq.opts.train_file

    corpus = os.path.join("data", corpus_name)
    # Define path to new file
    datafile = os.path.join(corpus, data_file)
    trainfile = os.path.join(corpus, train_file)
    save_dir = os.path.join("model", corpus_name)

    return corpus, corpus_name, datafile, trainfile, save_dir


if __name__ == '__main__':
    seq2seq, searcher = init_model(model)
    corpus, corpus_name, datafile, trainfile, save_dir = get_para_from_seq2seq(seq2seq)
    voc = init_voc(corpus, corpus_name, datafile, trainfile, save_dir)

    # Begin chatting (uncomment and run the following line to begin)
    evaluateInput(searcher, voc, max_length=MAX_LENGTH)
    # test_path = os.path.join(corpus, "test.txt")
    # res_path = re.sub(r'\.ml', r'\.txt', model)
    # evaluateFile(searcher, voc, test_path, res_path, max_length=MAX_LENGTH)
