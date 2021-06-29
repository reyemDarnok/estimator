#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import sys
import os
from math import nan

import numpy as np
import pandas as pd
import argcomplete
import tensorflow as tf

from argparse import ArgumentParser

from numpy import NaN


def main():
    print(sys.version)
    args = perform_argparse()
    model = get_model(args)
    if args.training_file is not None:
        train_model(model, args)
    if args.input_file is not None:
        parse_input(model, args)


def perform_argparse() -> argparse.Namespace:
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
    parser.add_argument('-b', '--borders', nargs='+',
                        help='the borders for the classifications. Default to [0.1, 1, 10]',
                        default=[0.1, 1, 10])
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.input_file == '-':
        args.input_file = sys.stdin
    return args


def get_model(args):
    if not args.force_retrain and os.path.isdir(os.path.join(args.permanency_folder, 'model')):
        model = tf.keras.models.load_model(os.path.join(args.permanency_folder, 'model'))
    else:
        training_dataset = read_file(args.training_file, args)
        train_dataset = training_dataset.sample(frac=0.8, random_state=0)
        test_dataset = training_dataset.drop(train_dataset.index)

        train_features = train_dataset
        train_features.pop('Result')

        normalizer = tf.keras.layers.experimental.preprocessing.Normalization()
        normalizer.adapt(np.array(train_features))

        model = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(args.borders) + 1),
            tf.keras.layers.Softmax()
        ])

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def train_model(model, args):
    training_dataset = read_file(args.training_file, args)
    train_dataset = training_dataset.sample(frac=0.8, random_state=0)
    test_dataset = training_dataset.drop(train_dataset.index)

    train_features = train_dataset
    test_features = test_dataset

    train_labels = train_features.pop('Result')
    test_labels = test_features.pop('Result')
    model.fit(train_features, train_labels, epochs=10)
    model.evaluate(test_features, test_labels)
    model.save(os.path.join(args.permanency_folder, 'model'))


def parse_input(model, args):
    input_data = read_file(args.input_file, args)
    if 'Result' in input_data:
        input_data.pop('Result')
    predictions = model.predict(input_data).flatten()
    input_data['Result'] = predictions
    input_data.to_csv(args.out_file, index=False)


def read_file(filename, args):
    dtypes = {'crop': 'string', 'Timing': 'string', 'TDS': 'string', 'TSCF': 'string', 'Scenario': 'string',
              'DT50': 'float64', 'KOC': 'float64', 'Freundlich': 'float64', 'Result': 'float64', 'Day': 'float64',
              'Month': 'float64'}
    dataframe = pd.read_csv(filename, dtype=dtypes, na_values=['Na'])
    dataframe.dropna(inplace=True)
    dataframe.crop = dataframe.crop.astype('category').cat.codes.astype('float')
    dataframe.Timing = dataframe.Timing.astype('category').cat.codes.astype('float')
    dataframe.TDS = dataframe.TDS.astype('category').cat.codes.astype('float')
    dataframe.TSCF = dataframe.TSCF.astype('category').cat.codes.astype('float')
    dataframe.Scenario = dataframe.Scenario.astype('category').cat.codes.astype('float')
    dataframe.Result = dataframe.Result.transform(categorize_function(args.borders))
    return dataframe


def categorize_function(borders):
    def categorize(num):
        if num < borders[0]:
            return 0
        if num >= borders[len(borders) - 1]:
            return len(borders)
        low = borders[0]
        for index, high in enumerate(borders, start=0):
            if high == low:
                continue
            if low <= num < high:
                return index
            low = high

    return categorize


if __name__ == '__main__':
    main()
