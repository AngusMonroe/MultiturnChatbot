
from utils.util import *
from model import *

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def evaluate(encoder, decoder, searcher, voc, sentence, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc, max_length):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


def evaluateFile(encoder, decoder, searcher, voc, input_path, output_path, max_length):
    input_file = open(input_path, 'r', encoding='utf8')
    output_file = open(output_path, 'w', encoding='utf8')
    for line in input_file.readlines():
        try:
            # Get input sentence
            sentence = line.split('\t')
            # Normalize sentence
            input_sentence = normalizeString(sentence[0])
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_line = ' '.join(output_words) + '\t' + sentence[1]
            output_file.write(output_line)

        except KeyError:
            print("Error: Encountered unknown word.")
    input_file.close()
    output_file.close()
