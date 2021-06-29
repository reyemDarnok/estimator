import os

import numpy as np
import tensorflow as tf

from estimator.io import read_file


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