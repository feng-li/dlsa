#! /usr/bin/env python3

import numpy as np
import pandas as pd
import time
from datetime import timedelta
import sys, os

import random
import pickle

from sklearn.linear_model import SGDClassifier

from utils import clean_airlinedata, get_dummy_keys
# from linereader import dopen
import string


tic0 = time.perf_counter()
##----------------------------------------------------------------------------------------
## Logistic Regression with Single machine SGD
##----------------------------------------------------------------------------------------

# file_path = ['~/running/data/' + str(year) + '.csv.bz2' for year in range(1987, 2008 + 1)]
# file_path = ['~/running/data_raw/xa' + str(letter) + '.csv.bz2' for letter in string.ascii_lowercase[1:21]]
file_path = ['~/running/data_raw/xa' + str(letter) + '.csv.bz2' for letter in string.ascii_lowercase[0:21]]

model_saved_file_name = '~/running/single_sgd_finalized_model_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime() + '.pkl'
dummy_set = "~/running/data_raw/dummy_set.pkl"
nBatches = 1000
nEpochs = 5
Y_name = 'ArrDelay'
loss="log"
penalty = 'none'
fit_intercept = True
verbose = True
n_jobs = -1 # Use all processors

SGD_model = SGDClassifier(fit_intercept=fit_intercept, verbose=verbose,
                          penalty=penalty, loss=loss, n_jobs=n_jobs)


# numeric_column_names = ['ArrDelay', 'DayofMonth', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
#                         'ActualElapsedTime', 'AirTime', 'DepDelay', 'Distance']

dummy_column_names = get_dummy_keys(dummy_set, fit_intercept)

# The main SGD looping
loop_counter = 0
for iEpoch in range(nEpochs):

    for file_number in range(len(file_path)):

        sample_df0 = clean_airlinedata(os.path.expanduser(file_path[file_number]),
                                       fit_intercept=fit_intercept)

        # Create an full-column empty DataFrame and resize current subset
        edf = pd.DataFrame(columns=list(set(dummy_column_names) - set(sample_df0.columns)))# empty df
        sample_df = sample_df0.append(edf, sort=True)
        del sample_df0
        sample_df.fillna(0, inplace = True) # Replace append-generated NaN with 0


        sample_df_size = sample_df.shape[0]
        total_idx = list(range(sample_df_size))
        random.shuffle(total_idx)

        x_train = sample_df.drop([Y_name], axis=1)
        y_train = sample_df[Y_name]
        classes=np.unique(y_train)

        for iBatch in range(nBatches):

            idx_curr_batch = total_idx[round(sample_df_size / nBatches * iBatch):
                                       round(sample_df_size / nBatches * (iBatch + 1))]


            SGD_model.partial_fit(x_train.iloc[idx_curr_batch, ],
                                  y_train.iloc[idx_curr_batch, ], classes=classes)


        print(str(iEpoch + 1) + '/' + str(nEpochs) + " Epochs:\t"
              + file_path[file_number] + '\tprocessed.\t'
              + str(file_number + 1) + '/' + str(len(file_path))
              + ' files done in this epoch.')

        loop_counter += 1
        time_elapsed = time.perf_counter() - tic0
        time_to_go = timedelta(seconds=time_elapsed / loop_counter * (len(file_path) * nEpochs - loop_counter))
        print('Time elapsed:\t' + str(timedelta(seconds=time_elapsed))
              + '.\tTime to go:\t' + str(time_to_go))


# tic = time.clock()
# parsedData = assembler.transform(data_sdf)
# time_parallelize = time.clock() - tic

# tic = time.clock()
# # Model configuration
# lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

# # Fit the model
# lrModel = lr.fit(parsedData)
# time_clusterrun = time.clock() - tic

# # Model fitted
# print(lrModel.intercept)
# print(lrModel.coefficients)

time_wallclock = time.perf_counter() - tic0

## Save model as file
pickle.dump(SGD_model, open(os.path.expanduser(model_saved_file_name), 'wb'))
# out = [sample_size, p, memsize, time_parallelize, time_clusterrun, time_wallclock]
# print(", ".join(format(x, "10.4f") for x in out))

## Load model
# with open(os.path.expanduser(model_saved_file_name), "rb") as f:
#     dump = pickle.load(f)
