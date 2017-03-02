"""
Model definition
"""
import json
import csv

import numpy as np
import tensorflow as tf
import pandas
import cv2

from keras.layers.core import Activation, Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

tf.python.control_flow_ops = tf

# Custom utilities and constants
from utils import IMAGE_SHAPE
from utils import flip, load_img, normalize, show_img, translate, perturb_brightness, add_shadow, do_flip
from utils import show_img, plot_angle_dist, plot_loss

# HYPER PARAMETERS
STEERING_OFFSET = 0.20  # Angle offset used for left/right camera images.
EPOCHS = 12             # Epochs used for training.
BATCH_SIZE = 256        # Batch sizes for training.
DATA_PATH = './data/'   # Location of csv file containing training dataset metadata.
LEARNING_RATE = 0.0001  # Adam's optimizer learning rate.
DROPOUT = 0.5           # Model's dropout percentage.

# LOADING METADATA
def data_path(p):
    """
    Prefixes correctly paths to look for training files.
    """
    return DATA_PATH + p.strip()

def load_img_from_csv_line(line):
    """
    Extracts an image for a given csv line. Randomly chooses
    between left, center or right image.

    Returns a tuple of image and respective steering angle.
    """
    steering_angle = float(line["steering"])

    side = np.random.randint(3)

    if side == 0:
        img_path = line["left"]
        steering_angle += STEERING_OFFSET
    elif side == 1:
        img_path = line["center"]
    else:
        img_path = line["right"]
        steering_angle -= STEERING_OFFSET

    return (load_img(data_path(img_path)), steering_angle)


def process_csv_line(line):
    """
    Given a csv line, it loads the image and performs some modifications
    with the purpose of equalizing steering angles distribution.
    """
    img, angle = load_img_from_csv_line(line)

    # image perturbations
    img, angle = translate(img , angle)
    img = perturb_brightness(img)
    img = add_shadow(img)
    if do_flip():
        img, angle = flip(img, angle)
    
    return img, angle

# data augmentation
def generate_data(metadata, batch_size = 32, bias = 1.0):
    """
    """

    samples = len(metadata)

    shuffle(metadata)

    batch_images = np.zeros( (batch_size, ) + IMAGE_SHAPE )
    batch_angles = np.zeros( batch_size )

    while 1:
        for batch_id in range(batch_size):
            csv_line_id = np.random.randint(samples)
            csv_line = metadata.iloc[csv_line_id]

            keep_sample = False

            while not keep_sample:
                img, angle = process_csv_line(csv_line)

                threshold = np.random.uniform()
                
                if (abs(angle) + bias) >= threshold:
                    keep_sample = True
                    
            batch_images[batch_id] = img
            batch_angles[batch_id] = angle

            angles.append(angle)

        yield shuffle(np.array(batch_images), np.array(batch_angles))


def build_model():
    """
    Defining the network architecture.
    """
    activation = 'relu'
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=IMAGE_SHAPE))
    model.add(Lambda(normalize))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), activation=activation))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), activation=activation))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), activation=activation))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation=activation))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1), activation=activation))
    model.add(Flatten())
    model.add(Dense(100, activation=activation))
    model.add(Dropout(DROPOUT))
    model.add(Dense(50, activation=activation))
    model.add(Dropout(DROPOUT))
    model.add(Dense(10, activation=activation))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))
    print(model.summary())
    return model


def persist_model(model, plot=False):
    """
    We take a model and persist it on disk.
    This method produces two files: model.json and model.h5

    Optionally, image model.png will be produced with the network
    architecture.
    """    
    model_json = model.to_json()
    with open('model.json', 'w') as modelFile:
        json.dump(model.to_json(), modelFile)
        modelFile.close()
    model.save('model.h5')

    if plot:
        plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)

angles = []

def train(model, metadata_train, metadata_val):
    """
    Performs the training of the given model, using
    training metadata to produce training data and 
    metadata validation to produce validation data.

    A tuple of lists is returned. Each list represents the loss
    at each epoch for training set and validation set respectively.
    """

    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse', metrics=['mean_squared_error'])

    train_samples = BATCH_SIZE * 40
    val_samples = BATCH_SIZE * 4

    loss_hist = []
    val_loss_hist = []

    for it in range(EPOCHS):

        bias = 1.0/(it + 1)
        epoch_angles = []
        training_results = model.fit_generator(
            generate_data(metadata_train, batch_size = BATCH_SIZE, bias=bias),
            validation_data=generate_data(metadata_val, batch_size = BATCH_SIZE),
            nb_val_samples=val_samples,
            samples_per_epoch=train_samples,
            nb_epoch=1,
            verbose=1)

        loss_hist.extend(training_results.history['loss'])
        val_loss_hist.extend(training_results.history['val_loss'])

    plot_angle_dist(angles, "Steering angle distribution")

    return loss_hist, val_loss_hist

if __name__ == '__main__':
    metadata = pandas.read_csv(data_path("driving_log.csv"), sep=',')

    # spliting metadata for training and validation
    metadata_train, metadata_val = train_test_split(metadata, test_size=0.2)

    # create model and train it.
    model = build_model()
    loss, val_loss = train(model, metadata_train, metadata_val)
    persist_model(model)

    # generate image containing plot of training and validation loss.
    plot_loss(loss, val_loss)

    # to avoid issues explained here: https://github.com/tensorflow/tensorflow/issues/3388
    import gc; gc.collect()





