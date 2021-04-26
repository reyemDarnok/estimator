import sys

import numpy as np
import pandas as pd

import os
import yaml
import glob

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from argparse import ArgumentParser

parser = ArgumentParser("Uses tensorflow to predict a single value")
parser.add_argument('-i', '--input-file', type=str,
                    help='the input csv file to read and predict from', default='input.csv')
parser.add_argument('-p', '--permanency-folder', type=str,
                    help='the folder to use for permanency saving model data', default='permanency')
parser.add_argument('-t', '--training-file', type=str,
                    help='the csv file to train the model. Leave empty to force model reuse', default=None)
parser.add_argument('-f', '--force-retrain', action='store_true',
                    help='force the program to retrain the model,'
                         ' even if there is an applicable model in the permanency folder')
args = parser.parse_args()

print(sys.version)
print(tf.__version__)


def model_available(description: pd.DataFrame) -> bool:
    for file in glob.glob(os.path.join(args.permanency_folder, '*.description')):
        file_description = yaml.load(file)
        headers = file_description['headers']
        if len(description.columns) == len(headers) and (description.columns == headers).all():
            return True
    return False


input_dataset = pd.read_csv(args.input_file)
if args.force_retrain or model_available(input_dataset):
    if args.training_file is None:
        raise ValueError('--training-file has to be specified'
                         ' or there has to be a trained model in --permanency-folder.'
                         'Neither was found')
    dataset = pd.read_csv(args.training_file)

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

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

    history = model.fit(
        train_features, train_labels,
        epochs=100,
        #    verbose=0,  # suppress logging
        validation_split=0.2  # Calculate validation results on 20% of training data
    )

test_predictions = model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Result]')
plt.ylabel('Predictions [Result]')
lims = [0, 800]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot([0, 800], [0, 800])
plt.show()

print(model.predict([[0.5, 0.5, 0.5, 0.5, 0.5]]))
