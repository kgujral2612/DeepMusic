#
# Data Preprocessing for GAN
# reference: https://github.com/mathigatti/midi2img
#

from midi2img import *
import os


def convert_midi_to_img(path):
    os.chdir(path)
    midiz = os.listdir()
    midis = []
    for midi in midiz:
        midis.append(path + '/' + midi)

    for midi in midis:
        try:
            midi2image(midi)
        except:
            pass