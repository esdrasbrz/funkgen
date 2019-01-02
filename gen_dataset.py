"""
Crawler used to generate dataset of Favela Funk
"""

from crawler import letrasmus
import collections
import config
import re
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer

def gen_sequences(corpus, tokenizer):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences

def main():
    corpus = letrasmus.scrap(config.OUTPUT_DATASET_FILE + '.txt', n_songs=config.NUM_SONGS).split('\n')

    print('generating sequences...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    print('total words: %d' % total_words)

    
    sequences = gen_sequences(corpus, tokenizer)
    print('number of sequences: %d' % len(sequences))

    pickle.dump(tokenizer, open(config.OUTPUT_DATASET_FILE + '.tokenizer.pkl', 'wb'))
    pickle.dump(sequences, open(config.OUTPUT_DATASET_FILE + '.sequences.pkl', 'wb'))
    print('done!')


if __name__ == '__main__':
    main()
