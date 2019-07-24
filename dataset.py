"""
Crawler used to generate dataset of Favela Funk
"""

from crawler import letrasmus
import collections
import re
import pickle
import numpy as np
import argparse


def _split_text(text):
    return re.findall(r"[\w']+|[.,!?;]", text)


def _generate_dict(text, min_word_frequency):
    words = _split_text(text)

    count = collections.Counter(words).most_common()
    word_indices = dict()
    ignored_words = set()

    for word, c in count:
        if c < min_word_frequency:
            ignored_words.add(word)
        else:
            word_indices[word] = len(word_indices)

    indices_word = dict(zip(word_indices.values(), word_indices.keys()))

    return word_indices, indices_word, ignored_words


def _gen_sequences(lines, wi, iw, ign, sequence_len, step=1):
    sequences = []
    next_word = []
    for line in lines:
        words = _split_text(line)

        # checks if line doesn't have a ignored word
        # create sequence only if number of words is greater than sequence_Len
        if len(set(words).intersection(ign)) == 0 and len(words) > sequence_len:
            for i in range(0, len(words) - sequence_len, step):
                seq = list(map(lambda w: wi[w], words[i:i + sequence_len]))

                sequences.append(np.array(seq))
                next_word.append(wi[words[i + sequence_len]])

    return np.array(sequences), np.array(next_word)


def generate(num_songs, output_dataset_fp, sequence_len=4, min_word_frequency=5, force_download=True):
    text_fp = output_dataset_fp + '.txt'
    if force_download:
        text = letrasmus.scrap(text_fp, n_songs=num_songs)
    else:
        with open(text_fp, 'r') as fin:
            text = fin.read()

    print('generating dict...')
    word_indices, indices_word, ignored_words = _generate_dict(text, min_word_frequency)
    pickle.dump(word_indices, open(output_dataset_fp + '.wi.pkl', 'wb'))
    pickle.dump(indices_word, open(output_dataset_fp + '.iw.pkl', 'wb'))
    pickle.dump(ignored_words, open(output_dataset_fp + '.ign.pkl', 'wb'))
    print('done!')

    print('generating sequences...')
    sequences, next_word = _gen_sequences(text.split('\n'), word_indices, indices_word, ignored_words, sequence_len)
    print('sequences shape: ', sequences.shape)

    np.save(output_dataset_fp + '.x.npy', sequences)
    np.save(output_dataset_fp + '.y.npy', next_word)
    print('done!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--nsongs', required=True, help='number of songs to scrap')
    ap.add_argument('-o', '--output', required=True, help='output file path')
    ap.add_argument('-s', '--seqlen', required=False, default=4, help='length of sequence')
    ap.add_argument('-w', '--minwordfreq', required=False, default=5, help='minimal word frequency')
    ap.add_argument('-f', '--force', required=False, default=True, help='force download')
    args = vars(ap.parse_args())

    def str2bool(s): return s.lower() in ['true', 't', '1']

    generate(
        num_songs=int(args['nsongs']),
        output_dataset_fp=args['output'],
        sequence_len=int(args['seqlen']),
        min_word_frequency=int(args['minwordfreq']),
        force_download=str2bool(args['force'])
    )
