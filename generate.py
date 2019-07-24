from model import FunkgenModel
import config
import pickle
import numpy as np


def main():
    # load dictionaries
    print('loading dicts...')
    wi = pickle.load(open(config.OUTPUT_DATASET_FILE + '.wi.pkl', 'rb'))
    iw = pickle.load(open(config.OUTPUT_DATASET_FILE + '.iw.pkl', 'rb'))
    ign = pickle.load(open(config.OUTPUT_DATASET_FILE + '.ign.pkl', 'rb'))
    print('done!')

    print('loading model...')
    model = FunkgenModel(wi, iw, ign, config.SEQUENCE_LEN)
    model.load_model_from_file(config.PREDICT_CHECKPOINT_FILEPATH)
    print('done!')

    while True:
        sequence = input('Seed sequence with %d words: ' % config.SEQUENCE_LEN).lower()
        sequence = sequence.split(' ')
        assert len(sequence) == config.SEQUENCE_LEN

        sequence_index = [] 
        for w in sequence:
            if w not in wi:
                print('The word %s is not in dict!' % w)
                sequence_index = None
                break
            else:
                sequence_index.append(wi[w])

        if sequence_index:
            sequence_index = np.array(sequence_index)

            temperature = float(input('Temperature (0 a 1): '))
            size = int(input('Size of generated sequence: '))

            lyrics = model.generate_lyrics(sequence_index, temperature=temperature, size=size)
            lyrics = ' '.join(lyrics)
            print(lyrics)

            fp = input('Output filepath (Blank to not save): ')
            if fp:
                with open(fp, 'w') as fout:
                    fout.write(lyrics)

        op = input('Generate another funk? (y/N) ').lower()
        if op != 'y':
            break


if __name__ == '__main__':
    main()
