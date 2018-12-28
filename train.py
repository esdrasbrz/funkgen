from model import FunkgenModel
import numpy as np
import config
import pickle



def main():
    # load dictionaries and data
    print('Loading datasets...')
    wi = pickle.load(open(config.OUTPUT_DATASET_FILE + '.wi.pkl', 'rb'))
    iw = pickle.load(open(config.OUTPUT_DATASET_FILE + '.iw.pkl', 'rb'))
    ign = pickle.load(open(config.OUTPUT_DATASET_FILE + '.ign.pkl', 'rb'))
    sequence = np.load(config.OUTPUT_DATASET_FILE + '.x.npy')
    next_word = np.load(config.OUTPUT_DATASET_FILE + '.y.npy')
    print('done!')

    model = FunkgenModel(wi, iw, ign, config.SEQUENCE_LEN)
    model.train(sequence, next_word, config.EPOCH_OUTPUT_FILEPATH, config.EPOCH_OUTPUT_FILEPATH)

if __name__ == '__main__':
    main()
