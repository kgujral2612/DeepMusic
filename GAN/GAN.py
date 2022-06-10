from DP import *

# this was used to convert midi files to images
# path = "/Users/kaushambigujral/Desktop/Grad/ML/DeepMusic/GAN/midi_images"
# convert_midi_to_img(path)

import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU, \
    Dropout
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from numpy import zeros, ones, vstack
from numpy.random import randn, randint
from IPython.display import clear_output


def access_images(img_list, path, length):
    pixels = []
    imgs = []
    for i in range(length):
        if 'png' in img_list[i]:
            try:
                img = Image.open(path + '\\' + img_list[i], 'r')
                img = img.convert('1')
                pix = np.array(img.getdata())
                pix = pix.astype('float32')
                pix /= 255.0
                pixels.append(pix.reshape(106, 106, 1))
                imgs.append(img)
            except Exception as e:
                print(e)
                pass
    return np.array(pixels), imgs


def show_image(pix_list):
    array = np.array(pix_list.reshape(106, 106), dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.show()


def define_discriminator(in_shape=(106, 106, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 53 * 53
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((53, 53, 128)))
    model.add(Dense(1024))
    model.add(Conv2DTranspose(1024, (4, 4), strides=(2, 2), padding='same'))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(Conv2D(1, (7, 7), padding='same', activation='sigmoid'))
    return model


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


def generate_real_samples(dataset, n_samples):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=51, n_batch=10):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss, g_loss))
            # if (i+1) % 10 == 0:
            #     summarize_performance(i, g_model, d_model, dataset, latent_dim)
            #     clear_output()


if __name__ == '__main__':
    # path = "C:\\Users\\Adam\\Repos\\DeepMusic\\GAN\\images"
    img_list = os.listdir(path)

    pixels, imgs = access_images(img_list, path, 200)
    latent_dim = 100
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    print(pixels.shape)
    train(g_model, d_model, gan_model, np.array(pixels), latent_dim)

    model = g_model
    latent_points = generate_latent_points(latent_dim, 1)
    X = g_model.predict(latent_points)

    array = np.array(X.reshape(106, 106), dtype=np.uint8)
    array *= 255
    new_image = Image.fromarray(array, 'L')
    new_image = new_image.save('composition.png')
