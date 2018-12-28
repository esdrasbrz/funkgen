"""
Based on
https://github.com/enriqueav/lstm_lyrics
"""

from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np

class FunkgenModel:
    def __init__(self, wi, iw, ign, sequence_len, percentage_test=5, \
                 batch_size=32, epochs=50, patience=5, dropout=0.2, cells=128):
        self.wi = wi
        self.iw = iw
        self.ign = ign
        self.sequence_len = sequence_len
        self.n_words = len(wi)
        self.percentage_test = percentage_test

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.dropout = dropout
        self.cells = cells

    def generator(self, sentence_list, next_word_list, batch_size, sequence_len, n_words):
        """
        Data generator for fit and evaluate
        """

        index = 0
        while True:
            x = np.zeros((batch_size, sequence_len, n_words), dtype=np.bool)
            y = np.zeros((batch_size, n_words), dtype=np.bool)
            for i in range(batch_size):
                for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                    x[i, t, w] = 1
                y[i, next_word_list[index % len(sentence_list)]] = 1
                index = index + 1

            yield x, y

    def shuffle_split_data(self, sentence, next_word):
        permutation = np.random.permutation(len(sentence))
        sentence = sentence[permutation]
        next_word = next_word[permutation]

        cut_index = int(len(sentence) * self.percentage_test / 100.)
        x_test, x_train = sentence[:cut_index], sentence[cut_index:]
        y_test, y_train = next_word[:cut_index], next_word[cut_index:]

        return (x_train, y_train), (x_test, y_test)

    def get_model(self):
        """
        Generates neural network model with Keras
        """

        model = Sequential() 
        model.add(Bidirectional(LSTM(self.cells), input_shape=(self.sequence_len, self.n_words)))
        if self.dropout > 0:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.n_words))
        model.add(Activation('softmax'))

        return model


    def generate_lyrics(self, seed_sequence, temperature=.5, size=10):
        # Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
        def sample(preds, temperature=1.0):
            # helper function to sample an index from a probability array
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)

        output = []

        # add to output the seed
        sequence = seed_sequence.tolist()
        for w in sequence:
            output.append(self.iw[w])

        for i in range(size-len(sequence)):
            x_pred = np.zeros((1, self.sequence_len, self.n_words))
            for t, w in enumerate(sequence):
                x_pred[0, t, w] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_word = self.iw[next_index]

            output.append(next_word)
            sequence = sequence[1:]
            sequence.append(next_index)

        return output

    def train(self, x, y, filepath, epoch_output_filepath):
        (x_train, y_train), (x_test, y_test) = self.shuffle_split_data(x, y)

        self.model = self.get_model()
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

        epoch_output_file = open(epoch_output_filepath, 'w')

        def on_epoch_end(epoch, logs):
            epoch_output_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

            # Randomly pick a seed sequence
            seed_index = np.random.randint(x_train.shape[0])
            seed = x_train[seed_index]

            for temperature in [0.2, 0.5, 0.8]:
                epoch_output_file.write('----- Temperature:' + str(temperature) + '\n')

                sentence = self.generate_lyrics(seed, temperature=temperature)
                epoch_output_file.write(' '.join(sentence))

                epoch_output_file.write('\n')

            epoch_output_file.write('='*80 + '\n')
            epoch_output_file.flush()

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True)
        print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
        early_stopping = EarlyStopping(monitor='val_acc', patience=self.patience)
        callbacks_list = [checkpoint, print_callback, early_stopping]

        self.model.fit_generator(self.generator(x_train, y_train, self.batch_size, self.sequence_len, self.n_words),
                                 steps_per_epoch=int(len(x_train) / self.batch_size) + 1,
                                 epochs=self.epochs,
                                 callbacks=callbacks_list,
                                 validation_data=self.generator(x_test, y_test, self.batch_size, self.sequence_len, self.n_words),
                                 validation_steps=int(len(x_test) / self.batch_size) + 1)

        epoch_output_file.close()
        return self.model
