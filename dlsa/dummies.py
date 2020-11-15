#! /usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import os
from collections import Counter


def dummy_factors_counts(pdf, dummy_columns):
    '''Function to count unique dummy factors for given dummy columns

    pdf: pandas data frame
    dummy_columns: list. Numeric or strings are both accepted.

    return: dict same as dummy columns
    '''
    # Check if current argument is numeric or string
    pdf_columns = pdf.columns.tolist()  # Fetch data frame header

    dummy_columns_isint = all(isinstance(item, int) for item in dummy_columns)
    if dummy_columns_isint:
        dummy_columns_names = [pdf_columns[i] for i in dummy_columns]
    else:
        dummy_columns_names = dummy_columns

    factor_counts = {}
    for i in dummy_columns_names:
        factor_counts[i] = (pdf[i]).value_counts().to_dict()

    return factor_counts


def cumsum_dicts(dict1, dict2):
    '''Merge two dictionaries and accumulate the sum for the same key where each dictionary
    containing sub-dictionaries with elements and counts.

    '''
    # If only one dict is supplied, do nothing.
    if len(dict1) == 0:
        dict_new = dict2
    elif len(dict2) == 0:
        dict_new = dict1
    else:
        dict_new = {}
        for i in dict1.keys():
            dict_new[i] = dict(Counter(dict1[i]) + Counter(dict2[i]))

    return dict_new


def select_dummy_factors(dummy_dict, keep_top, replace_with, pickle_file):
    '''Merge dummy key with frequency in the given file

    dummy_dict: dummy information in a dictionary format
    keep_top: list

    '''
    dummy_columns_name = list(dummy_dict)
    # nobs = sum(dummy_dict[dummy_columns_name[1]].values())

    factor_set = {}  # The full dummy sets
    factor_selected = {}  # Used dummy sets
    factor_dropped = {}  # Dropped dummy sets
    factor_selected_names = {}  # Final revised factors

    for i in range(len(dummy_columns_name)):

        column_i = dummy_columns_name[i]

        factor_set[column_i] = list((dummy_dict[column_i]).keys())

        factor_counts = list((dummy_dict[column_i]).values())
        factor_cumsum = np.cumsum(factor_counts)
        factor_cumpercent = factor_cumsum / factor_cumsum[-1]

        factor_selected_index = np.where(factor_cumpercent <= keep_top[i])
        factor_dropped_index = np.where(factor_cumpercent > keep_top[i])

        factor_selected[column_i] = list(
            np.array(factor_set[column_i])[factor_selected_index])

        factor_dropped[column_i] = list(
            np.array(factor_set[column_i])[factor_dropped_index])

        # Replace dropped dummies with indicators like `others`
        if len(factor_dropped_index[0]) == 0:
            factor_new = []
        else:
            factor_new = [replace_with]

        factor_new.extend(factor_selected[column_i])

        factor_selected_names[column_i] = [
            column_i + '_' + str(x) for x in factor_new
        ]

    dummy_info = {
        'factor_set': factor_set,
        'factor_selected': factor_selected,
        'factor_dropped': factor_dropped,
        'factor_selected_names': factor_selected_names
    }

    pickle.dump(dummy_info, open(os.path.expanduser(pickle_file), 'wb'))
    print("dummy_info saved in:\t" + pickle_file)

    return dummy_info


def select_dummy_factors_from_file(file, header, dummy_columns, keep_top,
                                   replace_with, pickle_file):
    '''Memory constrained algorithm to select dummy factors from a large file

    '''

    dummy_dict = {}
    buffer_num = 0
    with open(file) as f:
        while True:
            buffer = f.readlines(
                1024000)  # Returns *at most* 1024000 bytes, maybe less
            if len(buffer) == 0:
                break
            else:
                buffer_list = [x.strip().split(",") for x in buffer]

                buffer_num += 1
                if ((buffer_num == 1) and (header is True)):
                    buffer_header = buffer_list[0]
                    buffer_starts = 1
                else:
                    buffer_starts = 0

                buffer_pdf = pd.DataFrame(buffer_list[buffer_starts:])
                if header is True:
                    buffer_pdf.columns = buffer_header

                dummy_dict_new = dummy_factors_counts(buffer_pdf,
                                                      dummy_columns)

                dummy_dict = cumsum_dicts(dummy_dict, dummy_dict_new)

    dummy_info = select_dummy_factors(dummy_dict, keep_top, replace_with,
                                      pickle_file)
    return (dummy_info)


if __name__ == "__main__":

    # User settings
    file = os.path.expanduser("~/running/data/airdelay/airdelay_full.csv")
    header = True
    dummy_columns = [
        'Year', 'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'
    ]
    keep_top = [1, 1, 1, 0.8, 0.8, 0.8]
    replace_with = 'zzz_OTHERS'
    pickle_file = os.path.expanduser("~/running/data/airdelay/dummy_info.pkl")

    dummy_info = select_dummy_factors_from_file(file, header, dummy_columns,
                                                keep_top, replace_with,
                                                pickle_file)
