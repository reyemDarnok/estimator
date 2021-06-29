import argparse
import sys
from argparse import ArgumentParser

import argcomplete


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
