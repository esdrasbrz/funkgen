"""
Crawler used to generate dataset of Favela Funk
"""

from crawler import letrasmus
import collections
import config
import re
import pickle

def generate_dict(text):
    words = re.findall(r"[\w']+|[.,!?;]", text)

    count = collections.Counter(words).most_common()
    word_indices = dict()
    ignored_words = set()

    for word, c in count:
        if c < config.MIN_WORD_FRQUENCY:
            ignored_words.add(word)
        else:
            word_indices[word] = len(word_indices)

    indices_word = dict(zip(word_indices.values(), word_indices.keys()))

    return word_indices, indices_word, ignored_words

def main():
    text = letrasmus.scrap(config.OUTPUT_DATASET_FILE + '.txt', n_songs=config.NUM_SONGS)

    print('generating dict...')
    word_indices, indices_word, ignored_words = generate_dict(text)
    pickle.dump(word_indices, open(config.OUTPUT_DATASET_FILE + '.wi.pkl', 'wb'))
    pickle.dump(indices_word, open(config.OUTPUT_DATASET_FILE + '.iw.pkl', 'wb'))
    pickle.dump(ignored_words, open(config.OUTPUT_DATASET_FILE + '.ign.pkl', 'wb'))
    print('done!')

if __name__ == '__main__':
    main()
