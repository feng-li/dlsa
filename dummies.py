#! /usr/bin/env

import pickle
import pandas as pd
import os

def select_dummy_factors(pdf, dummy_columns, kep_top):
    '''Merge dummy key with frequency in the given file

    '''
    nobs = pdf.shape[0]

    factor_set = {}
    factor_selected = {}
    factor_dropped = {}
    factor_selected_names = []

    for i in range(len(dummy_columns)):

        factor_counts = (pdf[dummy_columns[i]]).value_counts()
        factor_cum = factor_counts.cumsum() / nobs

        factor_selected[dummy_columns[i]] = sorted(list(factor_counts.index[factor_cum <= keep_top[i]]))
        factor_dropped[dummy_columns[i]] = sorted(list(factor_counts.index[factor_cum > keep_top[i]]))
        factor_set[dummy_columns[i]] = sorted(list(factor_counts.index))
        factor_selected_names.extend([dummy_columns[i] + '_' + str(x) for x in factor_selected[dummy_columns[i]]])


    dummy_info = {'factor_set': factor_set,
                  'factor_selected': factor_selected,
                  'factor_dropped': factor_dropped,
                  'factor_selected_names': factor_selected_names}

    pickle.dump(dummy_info, open(os.path.expanduser('~/running/data_raw/dummy_info.pkl'), 'wb'))

    return dummy_info


# pdf = pd.read_csv(os.path.expanduser("~/running/data_raw/allfile.csv.bz2"),
#                   error_bad_lines=False,
#                   usecols = [1,2,3,4,5,6,7,8,11,13,14,15,16,17,18],
#                   engine='c') # The C engine is faster
# dummy_info = select_dummy_factors(pdf,
#                                   dummy_columns=['Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
#                                   kep_top= [1, 1, 0.95, 0.95, 0.95])
#
