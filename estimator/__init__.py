from estimator.io import parse_input
from estimator.project_argparse import perform_argparse

import sys


def get_model():
    pass


def train_model():
    pass


def main():
    print(sys.version)
    args = perform_argparse()
    from estimator.model_functions import get_model, train_model
    model = get_model(args)
    if args.training_file is not None:
        train_model(model, args)
    if args.input_file is not None:
        parse_input(model, args)
