#! /usr/bin/env python3

import pickle
import pandas as pd
import os

def select_dummy_factors(pdf, dummy_columns, kep_top, replace_with="00_OTHERS"):
    '''Merge dummy key with frequency in the given file

    '''
    nobs = pdf.shape[0]

    factor_set = {}
    factor_selected = {}
    factor_dropped = {}
    factor_selected_names = {}

    for i in range(len(dummy_columns)):

        factor_counts = (pdf[dummy_columns[i]]).value_counts()
        factor_cum = factor_counts.cumsum() / nobs

        factor_selected[dummy_columns[i]] = sorted(list(factor_counts.index[factor_cum <= keep_top[i]]))
        factor_dropped[dummy_columns[i]] = sorted(list(factor_counts.index[factor_cum > keep_top[i]]))
        factor_set[dummy_columns[i]] = sorted(list(factor_counts.index))

        if len(factor_dropped[dummy_columns[i]]) == 0:
            factor_new = []
        else:
            factor_new = replace_with
        factor_new.extend(factor_selected[dummy_columns[i]])

        factor_selected_names[dummy_columns[i]] = [dummy_columns[i] + '_' + str(x) for x in factor_new]


    dummy_info = {'factor_set': factor_set,
                  'factor_selected': factor_selected,
                  'factor_dropped': factor_dropped,
                  'factor_selected_names': factor_selected_names}

    pickle.dump(dummy_info, open(os.path.expanduser('~/running/data_raw/dummy_info.pkl'), 'wb'))

    return dummy_info


pdf = pd.read_csv(os.path.expanduser("~/running/data_raw/allfile.csv.bz2"),
                  error_bad_lines=False,
                  usecols = [1, 3, 8, 16, 17],
                  engine='c') # The C engine is faster
dummy_info = select_dummy_factors(pdf,
                                  dummy_columns=['Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
                                  kep_top= [1, 1, 0.9, 0.9, 0.9],
                                  replace_with='00_OTHERS'
)
