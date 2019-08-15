#! /usr/bin/env python3

import pickle
import pandas as pd
import os

def select_dummy_factors(pdf, dummy_columns, keep_top, replace_with, pickle_file):
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
            factor_new = [replace_with]

        factor_new.extend(factor_selected[dummy_columns[i]])

        factor_selected_names[dummy_columns[i]] = [dummy_columns[i] + '_' + str(x) for x in factor_new]


    dummy_info = {'factor_set': factor_set,
                  'factor_selected': factor_selected,
                  'factor_dropped': factor_dropped,
                  'factor_selected_names': factor_selected_names}

    pickle.dump(dummy_info, open(os.path.expanduser(pickle_file), 'wb'))
    print("dummy_info saved in:\t" + pickle_file)

    return dummy_info



if __name__ == "__main__":
    pdf = pd.read_csv(os.path.expanduser("~/running/data_raw/dummies.csv.bz2"),
                      error_bad_lines=False,
                      engine='c') # The C engine is faster
    dummy_info = select_dummy_factors(
        pdf,
        dummy_columns=['Year', 'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
        keep_top= [1, 1, 1, 0.8, 0.9, 0.9],
        replace_with='00_OTHERS',
        pickle_file='~/running/data_raw/dummy_info_latest.pkl')
