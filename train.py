from model import FunkgenModel
import numpy as np
import config
import pickle



def main():
    # load dictionaries and data
    print('Loading datasets...')
    meta = pickle.load(open(config.OUTPUT_DATASET_FILE + '.meta.pkl', 'rb'))
    x_train = np.load(config.OUTPUT_DATASET_FILE + '.train.x.npy')
    y_train = np.load(config.OUTPUT_DATASET_FILE + '.train.y.npy')
    x_test = np.load(config.OUTPUT_DATASET_FILE + '.test.x.npy')
    y_test = np.load(config.OUTPUT_DATASET_FILE + '.test.y.npy')
    print('done!')

    model = FunkgenModel(meta)
    model.train(x_train, y_train,
                x_test, y_test,
                config.EPOCH_CHECKPOINT_FILEPATH, config.EPOCH_OUTPUT_FILEPATH)

if __name__ == '__main__':
    main()
