#
# Data Preprocessing for GAN
# reference: https://github.com/mathigatti/midi2img
#

from midi2img import *
from PIL import Image
import os


def convert_midi_to_img(midi_path, img_path):
    print("Converting midi files to png ...")
    os.chdir(midi_path)
    midiz = os.listdir()
    midis = []
    for midi in midiz:
        midis.append(midi_path + '\\' + midi)

    for midi in midis:
        try:
            midi2image(midi, img_path)
        except Exception as e:
            print(e)
            pass

# Resizes images to 106x106 pixels
# Images need to be 106x106 pixels to work with GAN.py
def resize_imgs(path):
    print("Resizing images ...")
    os.chdir(path)
    ls = os.listdir()
    imgs = []

    for img_path in ls:
        imgs.append(path + '\\' + img_path)
    
    for img_path in imgs:
        if 'png' in img_path:
            try:
                shape = (106, 106)
                img = Image.open(img_path)
                img = img.resize(shape, Image.ANTIALIAS)
                img.save(img_path)
            except Exception as e:
                print(e)



if __name__ == '__main__':
    # midi_path = 'C:\\Users\\Adam\\Repos\\DeepMusic\\piano_songs'
    # img_path = 'C:\\Users\\Adam\\Repos\\DeepMusic\\GAN\\images'
    convert_midi_to_img(midi_path, img_path)
    resize_imgs(img_path)