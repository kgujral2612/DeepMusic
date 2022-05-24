
from DP import *

notes = read_files()
print(notes)

training_data = create_training_data(notes, len(set(notes)))
network_input, network_output = training_data
print(network_input.shape)
print(network_output.shape)

