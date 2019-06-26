"""
Based on
https://github.com/enriqueav/lstm_lyrics
"""

from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
import numpy as np

class FunkgenModel:
    def __init__(self, meta,\
                 batch_size=256, epochs=50, patience=5, dropout=0.2, cells=128):
        self.tokenizer = meta['tokenizer']
        self.n_words = meta['n_words']
        self.sequence_len = meta['sequence_len']
        self.reverse_word_map = dict(map(reversed, self.tokenizer.word_index.items()))

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
        n_sentences = len(sentence_list)
        while True:
            x = np.zeros((batch_size, sequence_len-1), dtype=np.bool)
            y = np.zeros((batch_size, n_words), dtype=np.bool)
            for i in range(batch_size):
                for t, tok in enumerate(sentence_list[index % n_sentences]):
                    x[i, t] = tok
                y[i, next_word_list[index % n_sentences].astype(np.bool)] = 1
                index = index + 1

            yield x, y

    def get_model(self):
        """
        Generates neural network model with Keras
        """

        model = Sequential() 
        model.add(Embedding(self.n_words, 10, input_length=self.sequence_len-1))
        model.add(Bidirectional(LSTM(self.cells)))
        if self.dropout > 0:
            model.add(Dropout(self.dropout))
        model.add(Dense(self.n_words))
        model.add(Activation('softmax'))

        return model

    def load_model_from_file(self, filepath):
        self.model = self.get_model()
        self.model.load_weights(filepath)
        self.model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    def generate_lyrics(self, seed_sequence, temperature=.5, size=40):
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
            if w != 0:
                output.append(self.reverse_word_map[w])

        initial_len = len(output)
        for i in range(size-initial_len):
            x_pred = np.zeros((1, self.sequence_len-1))
            for i, w in enumerate(reversed(output[-(self.sequence_len-1):])):
                x_pred[0, self.sequence_len-2-i] = self.tokenizer.word_index[w]

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_word = self.reverse_word_map[next_index]

            output.append(next_word)

        return output

    def train(self, x_train, y_train, x_test, y_test, filepath, epoch_output_filepath):
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
