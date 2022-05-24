#
# Data Preprocessing for LSTM
# reference: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
#

from music21 import *
import glob
import numpy as np
from tensorflow.keras import utils


# keeping this for learning purposes
def read_file(path):
    midi_file = converter.parse(path)
    components = []
    for element in midi_file.recurse():
        components.append(element)
    return components


def read_files():
    notes = []
    for path in glob.glob("piano_songs/*.mid"):
        midi_file = converter.parse(path)
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi_file)
        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:  # file has notes in a flat structure
            notes_to_parse = midi_file.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
        return notes


# notes: these were read from the midi files
# n_vocab: the vocabulary of notes known to this code
def create_training_data(notes, n_vocab, sequence_length=100):
    # Step-1: Map pitch names to integers
    note_to_int = dict((note_val, id) for id, note_val in enumerate(sorted(set(notes))))

    # Step-2: create input sequences and the corresponding outputs
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Step-3: reshape and normalize
    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    network_input = network_input / n_vocab
    network_output = utils.to_categorical(network_output)
    return network_input, network_output

