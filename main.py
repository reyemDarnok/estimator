#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

import sys

import numpy as np
import pandas as pd

import os
import yaml
import glob
import hashlib
import uuid

from typing import Optional
import matplotlib.pyplot as plt


from argparse import ArgumentParser
argcomplete_available = False

try:
    import argcomplete
    argcomplete_available = True
except ModuleNotFoundError:
    pass


parser = ArgumentParser("Uses tensorflow to predict a single value")
parser.add_argument('-i', '--input-file', type=str,
                    help='the input csv file to read and predict from')
parser.add_argument('-p', '--permanency-folder', type=str,
                    help='the folder to use for permanency saving model data', default='permanency')
parser.add_argument('-t', '--training-file', type=str,
                    help='the csv file to train the model. Leave empty to force model reuse')
parser.add_argument('-f', '--force-retrain', action='store_true',
                    help='force the program to retrain the model,'
                         ' even if there is an applicable model in the permanency folder')
parser.add_argument('-o', '--out-file', type=str,
                    help='the file to write the predictions to. '
                         'Has no effect if -i is not specified', default=sys.stdout)
if argcomplete_available:
    argcomplete.autocomplete(parser)
args = parser.parse_args()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(sys.version)
print(tf.__version__)

if not args.force_retrain and os.path.isdir(os.path.join(args.permanency_folder, 'model')):
    model = tf.keras.models.load_model(os.path.join(args.permanency_folder, 'model'))
else:
    if args.training_file is None:
        sys.exit("No available model and no training file - one of these must be available")

    training_dataset = pd.read_csv(args.training_file)
    train_dataset = training_dataset.sample(frac=0.8, random_state=0)
    test_dataset = training_dataset.drop(train_dataset.index)

    train_features = train_dataset
    test_features = test_dataset

    train_labels = train_features.pop('Result')
    test_labels = test_features.pop('Result')

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))

    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(units=1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error')
    model.fit(train_features, train_labels, epochs=10)
    model.save(os.path.join(args.permanency_folder, 'model'))

if args.input_file is not None:
    input_data = pd.read_csv(args.input_file)
    predictions = model.predict(input_data).flatten()
    with open(args.out_file, 'w') as output_file:
        output_file.write(predictions)

