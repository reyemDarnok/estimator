import pandas as pd


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
    if 'Result' in input_data:
        input_data.pop('Result')
    predictions = model.predict(input_data).flatten()
    input_data['Result'] = predictions
    input_data.to_csv(args.out_file, index=False)
