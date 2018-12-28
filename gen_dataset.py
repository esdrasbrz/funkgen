"""
Crawler used to generate dataset of Favela Funk
"""

from crawler import letrasmus
import collections
import config
import re
import pickle
import numpy as np

def split_text(text):
    return re.findall(r"[\w']+|[.,!?;]", text)

def generate_dict(text):
    words = split_text(text)

    count = collections.Counter(words).most_common()
    word_indices = dict()
    ignored_words = set()

    for word, c in count:
        if c < config.MIN_WORD_FREQUENCY:
            ignored_words.add(word)
        else:
            word_indices[word] = len(word_indices)

    indices_word = dict(zip(word_indices.values(), word_indices.keys()))

    return word_indices, indices_word, ignored_words

def gen_sequences(lines, wi, iw, ign, sequence_len, step=1):
    sequences = []
    next_word = []
    for line in lines:
        words = split_text(line)

        # checks if line doesn't have a ignored word
        # create sequence only if number of words is greater than sequence_Len
        if len(set(words).intersection(ign)) == 0 and len(words) > sequence_len:
            for i in range(0, len(words) - sequence_len, step):
                seq = list(map(lambda w: wi[w], words[i:i + sequence_len]))

                sequences.append(np.array(seq))
                next_word.append(wi[words[i+sequence_len]])

    return np.array(sequences), np.array(next_word)

def main():
    text = letrasmus.scrap(config.OUTPUT_DATASET_FILE + '.txt', n_songs=config.NUM_SONGS)

    print('generating dict...')
    word_indices, indices_word, ignored_words = generate_dict(text)
    pickle.dump(word_indices, open(config.OUTPUT_DATASET_FILE + '.wi.pkl', 'wb'))
    pickle.dump(indices_word, open(config.OUTPUT_DATASET_FILE + '.iw.pkl', 'wb'))
    pickle.dump(ignored_words, open(config.OUTPUT_DATASET_FILE + '.ign.pkl', 'wb'))
    print('done!')

    print('generating sequences...')
    sequences, next_word = gen_sequences(text.split('\n'), word_indices, indices_word, ignored_words, config.SEQUENCE_LEN)
    print('sequences shape: ', sequences.shape)

    np.save(config.OUTPUT_DATASET_FILE + '.x.npy', sequences)
    np.save(config.OUTPUT_DATASET_FILE + '.y.npy', next_word)
    print('done!')


if __name__ == '__main__':
    main()
