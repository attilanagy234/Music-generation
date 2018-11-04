import prepare_data as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

'''
This file is responsible for constructing and training the network
Weights are stored in .hdf5 files
'''

def build_network(net_input, vocab_size):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(net_input.shape[1], net_input.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, net_input, net_output):
    filepath = "weights-{epoch:02d}-{loss:.5f}-test.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose =0,
        save_best_only=True,
        mode='min')

    callbacks_list = [checkpoint]

    model.fit(net_input, net_output,
              epochs = 200,
              batch_size=64,
              callbacks=callbacks_list)

def train_net():
    notes = pd.convert_from_midi()

    vocab = len(set(notes))

    net_input, net_output = pd.generate_sequence(vocab, notes)

    model = build_network(net_input, vocab)

    train(model, net_input, net_output)


if __name__ == '__main__':
    train_net()