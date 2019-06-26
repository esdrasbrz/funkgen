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
from keras.preprocessing.sequence import pad_sequences
import keras.utils as ku 

def gen_sequences(corpus, tokenizer):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    return input_sequences

def preprocess(train_sequences, test_sequences, n_words):
    sequence_len = max([len(x) for x in train_sequences + test_sequences])
    train_sequences = np.array(pad_sequences(train_sequences, maxlen=sequence_len, padding='pre'))
    test_sequences = np.array(pad_sequences(test_sequences, maxlen=sequence_len, padding='pre'))

    x_train, y_train = train_sequences[:,:-1], train_sequences[:,-1]
    y_train = ku.to_categorical(y_train, num_classes=n_words)

    x_test, y_test = test_sequences[:,:-1], test_sequences[:,-1]
    y_test = ku.to_categorical(y_test, num_classes=n_words)

    return (x_train, y_train), (x_test, y_test), sequence_len

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

    (x_train, y_train), (x_test, y_test), sequence_len = preprocess(train_sequences, test_sequences, total_words)

    meta = {
        'sequence_len': sequence_len,
        'n_words': total_words,
        'tokenizer': tokenizer
    }

    pickle.dump(meta, open(config.OUTPUT_DATASET_FILE + '.meta.pkl', 'wb'))
    np.save(config.OUTPUT_DATASET_FILE + '.train.x.npy', x_train)
    np.save(config.OUTPUT_DATASET_FILE + '.train.y.npy', y_train)
    np.save(config.OUTPUT_DATASET_FILE + '.test.x.npy', x_test)
    np.save(config.OUTPUT_DATASET_FILE + '.test.y.npy', y_test)
    print('done!')


if __name__ == '__main__':
    main()
