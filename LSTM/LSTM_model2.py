from music21 import *
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DP import *


notes = read_files()
#print(notes)

n_vocab = len(set(notes))
training_data = create_training_data(notes, n_vocab)
network_input, network_output = training_data
print(network_input.shape)
print(network_output.shape)

model = keras.Sequential()

model.add(tf.keras.layers.LSTM(
	256,
	input_shape=(network_input.shape[1], network_input.shape[2]),
	return_sequences=True
))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.LSTM(256))
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(n_vocab))
model.add(tf.keras.layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

checkpoint = tf.keras.callbacks.ModelCheckpoint(
	filepath, monitor='loss', 
	verbose=0,        
	save_best_only=True,        
	mode='min'
) 
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)



