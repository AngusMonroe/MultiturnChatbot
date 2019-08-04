# -*- coding:utf-8 -*-
import codecs
import csv
import os
import re
import json

corpus_name = "duconv"
corpus = os.path.join("data", corpus_name)

# Define path to new file
datafile = os.path.join(corpus, 'formatted_dialog.txt')
# origin_file_list = ['train.txt', 'dev.txt', 'test_1.txt', 'test_2.txt', 'test.txt']
origin_file_list = ['train.txt', 'dev.txt']


def loadConversations(fileName, dics):
    f = open(fileName, 'r', encoding='utf-8')
    pairs = []
    for line in f.readlines():
        dic = json.loads(line)
        dics.append(dic)
        pairs = extractSentencePairs(pairs, dic)
    f.close()
    return dics, pairs


def extractSentencePairs(pairs, dic):
    if dic['conversation']:
        for i in range(len(dic['conversation'])):
            if not i:
                continue
            pair = []
            pair.append(dic['conversation'][i - 1])
            pair.append(dic['conversation'][i])
            pairs.append(pair)
    return pairs


delimiter = '\t'
# Unescape the delimiter
# delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Load lines and process conversations

dicts = []
# data_file = open(datafile, 'w', encoding='utf-8')
with open(datafile, 'w', encoding='utf-8') as formatted_file:
    wr = csv.writer(formatted_file, delimiter=delimiter, lineterminator='\n')
    for file in origin_file_list:
        print("Processing " + file + "...")
        dicts, pairs = loadConversations(os.path.join(corpus, file), dicts)
        file_name = re.sub(r'\.txt', '.dat', os.path.join(corpus, file))
        with open(file_name, 'w', encoding='utf-8') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
            for pair in pairs:
                writer.writerow(pair)
                wr.writerow(pair)
                # data_file.write(pair[0] + '\t' + pair[1] + '\n')
# data_file.close()

print('Done!')
