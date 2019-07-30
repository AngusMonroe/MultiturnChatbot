# -*- coding:utf-8 -*-
import glove
import torch
import numpy as np

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown-keyword token


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.index2emb = []
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.num_words = 4  # Count default tokens

        for word in keep_words:
            self.addWord(word)

    def getEmb(self, matrix):
        for i in self.index2word.keys():
            if self.index2word[i] in matrix.itos:
                idx = matrix.stoi[self.index2word[i]]
                self.index2emb.append(matrix.vectors.numpy()[idx])
            else:
                self.index2emb.append(np.zeros(300))



class GloveVocabBuilder(object):

    def __init__(self, path_glove):
        self.vec = None
        self.vocab = None
        self.path_glove = path_glove

    def get_word_index(self, padding_marker='PAD', unknown_marker='UNK',):
        print(self.path_glove)
        instance = glove.Glove.load_stanford(self.path_glove)
        print(type(instance.word_vectors))
        _vocab = instance.dictionary
        _vec = torch.FloatTensor(instance.word_vectors)
        # _vocab, _vec = torchwordemb.load_glove_text(self.path_glove)
        # print(_vec)
        vocab = {padding_marker: 0, unknown_marker: 1}
        print('loading...')
        for tkn, indx in _vocab.items():
            vocab[tkn] = indx + 2
        print('ok')
        vec_2 = torch.zeros((2, _vec.size(1)))
        vec_2[1].normal_()
        self.vec = torch.cat((vec_2, _vec))
        self.vocab = vocab
        return self.vocab, self.vec

