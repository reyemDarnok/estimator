from collections import Counter

import pandas as pd
import numpy as np

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


def parse_input(model, args):
    input_data = read_file(args.input_file, args)
    result = None
    if 'Result' in input_data:
        result = input_data.pop('Result')
    predictions = model.predict(input_data)
    if result is not None:
        input_data['Result'] = result
    low = args.borders[0]
    column_names = []
    for index, high in enumerate(args.borders):
        if low == high:
            column_names.append(f'x < {low}')
            continue
        column_names.append(f'{low} <= x < {high}')
        low = high
    column_names.append(f'{args.borders[len(args.borders) - 1]} <= x')
    predictions_df = pd.DataFrame(data=predictions, columns=column_names)
    input_data = pd.merge(input_data, predictions_df, left_index=True, right_index=True)
    input_data.to_csv(args.out_file, index=False)
