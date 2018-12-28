# Number of songs to scrap (-1 to get all)
NUM_SONGS = 100
OUTPUT_DATASET_FILE = 'dataset/funk-top-100'
MIN_WORD_FREQUENCY = 0
SEQUENCE_LEN = 4

EPOCH_CHECKPOINT_FILEPATH = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-songs%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.hdf5" \
    % (NUM_SONGS, SEQUENCE_LEN, MIN_WORD_FREQUENCY)
EPOCH_OUTPUT_FILEPATH = './output/train-generated-lyrics.txt'
