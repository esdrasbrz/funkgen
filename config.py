# Number of songs to scrap (-1 to get all)
NUM_SONGS = 1000
OUTPUT_DATASET_FILE = 'dataset/funk-top-1000'
MIN_WORD_FREQUENCY = 5
SEQUENCE_LEN = 4

EPOCH_CHECKPOINT_FILEPATH = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-songs%d-sequence%d-minfreq%d-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}.hdf5" \
    % (NUM_SONGS, SEQUENCE_LEN, MIN_WORD_FREQUENCY)
EPOCH_OUTPUT_FILEPATH = './output/train-generated-lyrics.txt'

PREDICT_CHECKPOINT_FILEPATH = './checkpoints/LSTM_LYRICS-epoch041-songs1000-sequence4-minfreq5-loss0.1686-acc0.9527-val_loss2.7841-val_acc0.7180.hdf5'
