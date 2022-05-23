#
# Parsing a midi file
#

from music21 import *
import glob


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
