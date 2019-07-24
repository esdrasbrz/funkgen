from model import FunkgenModel
import numpy as np
import pickle
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='dataset file path')
    ap.add_argument('-c', '--checkpoint', required=True, help='checkpoints file path')
    ap.add_argument('-o', '--output', required=True, help='output file path')
    ap.add_argument('-s', '--seqlen', required=False, default=4, help='length of sequence')
    args = vars(ap.parse_args())

    # load dictionaries and data
    print('Loading datasets...')
    wi = pickle.load(open(args['dataset'] + '.wi.pkl', 'rb'))
    iw = pickle.load(open(args['dataset'] + '.iw.pkl', 'rb'))
    ign = pickle.load(open(args['dataset'] + '.ign.pkl', 'rb'))
    sequence = np.load(args['dataset'] + '.x.npy')
    next_word = np.load(args['dataset'] + '.y.npy')
    print('done!')

    model = FunkgenModel(wi, iw, ign, int(args['seqlen']))
    model.train(sequence, next_word, args['checkpoint'], args['output'])


if __name__ == '__main__':
    main()
