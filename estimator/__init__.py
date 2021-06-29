from estimator.io import parse_input
from estimator.model_functions import get_model, train_model
from estimator.project_argparse import perform_argparse

import sys


def main():
    print(sys.version)
    args = perform_argparse()
    model = get_model(args)
    if args.training_file is not None:
        train_model(model, args)
    if args.input_file is not None:
        parse_input(model, args)


if __name__ == '__main__':
    main()
