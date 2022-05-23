from Helpers.MidiHelper import *

if __name__ == "__main__":
    a = [10, 20, 30]
    b = [10, 20, 30]
    c = a + b
    print(c)
    notes = read_files()
    training_data = create_training_data(notes, len(set(notes)))
    print(notes)
