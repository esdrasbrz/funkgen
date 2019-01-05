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
    train_corpus, test_corpus = letrasmus.scrap(config.OUTPUT_DATASET_FILE + '.train.txt', \
                                                config.OUTPUT_DATASET_FILE + '.test.txt', \
                                                n_songs=config.NUM_SONGS)

    train_corpus = train_corpus.split('\n')
    test_corpus = test_corpus.split('\n')

    print('generating sequences...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_corpus + test_corpus)
    total_words = len(tokenizer.word_index) + 1
    print('total words: %d' % total_words)

    
    train_sequences = gen_sequences(train_corpus, tokenizer)
    test_sequences = gen_sequences(test_corpus, tokenizer)
    print('number of train sequences: %d' % len(train_sequences))
    print('number of test sequences: %d' % len(test_sequences))

    pickle.dump(tokenizer, open(config.OUTPUT_DATASET_FILE + '.tokenizer.pkl', 'wb'))
    pickle.dump(train_sequences, open(config.OUTPUT_DATASET_FILE + '.train.sequences.pkl', 'wb'))
    pickle.dump(test_sequences, open(config.OUTPUT_DATASET_FILE + '.test.sequences.pkl', 'wb'))
    print('done!')


if __name__ == '__main__':
    main()
