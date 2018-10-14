from music21 import note, chord, instrument, converter
import glob
import pickle
import numpy
from keras.utils import np_utils

'''
This file is responsible for preparing the data for training, 
by encoding the MIDI formatted music samples to .txt format.
Every .mid file in the "midi_songs" folder will be encoded to notes.txt
'''


def convert_from_midi():
    notes = []

    for my_file in glob.glob("midi_data/*.mid"):
        midi = converter.parse(my_file)
        notes_to_parse = None

        print("Currently parsing %s" % my_file)

        # if there are instruments in the midi file:
        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()
        except IOError:
            print("The file has flat structured notes")
            notes_to_parse = midi.flat.notes
        for i in notes_to_parse:
            if isinstance(i, note.Note):
                notes.append(str(i.pitch))
            elif isinstance(i, chord.Chord):
                notes.append('.'.join(str(x) for x in i.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


'''
Having the sequential data in a text file, 
we have to create sequences that can be fed into the network.
The network determines an output note after examining 'seq_length' number of notes
'''


def generate_sequence(vocab, notes):
    net_input = []
    net_output = []
    # sequence_length is going to be determined through hyperparameter optimisation
    seq_length = 50

    # mapping pitchnames to integers
    pitch_names = sorted(set(element for element in notes))
    encoded_notes = dict((note, num) for num, note in enumerate(pitch_names))

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - seq_length):
        seq_in = notes[i:i + seq_length]
        seq_out = notes[i + seq_length]
        net_input.append([encoded_notes[char] for char in seq_in])
        net_output.append(encoded_notes[seq_out])

    n_patterns = len(net_input)

    # reshape the input into a format compatible with the LSTM layers
    net_input = numpy.reshape(net_input, (n_patterns, seq_length, 1))

    # normalize input
    net_input = net_input / float(vocab)
    net_output = np_utils.to_categorical(net_output)

    print("Net_input is:", net_input)
    print("Net_output is:", net_output)

    return net_input, net_output
