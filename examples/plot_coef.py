#! /usr/bin/env python3

import pickle
import os
import pandas as pd
import numpy as np

sgd = "~/running/logistic_sgd_model_2019-07-29-11:12:04.pkl"
dlsa = "~/running/logistic_dlsa_model_2019-08-05-01:12:30.pkl"

est_sgd = pickle.load(open(os.path.expanduser(sgd), "rb"))
est_dlsa = pd.read_pickle(os.path.expanduser(dlsa))

est_dlsa_key = list(est_dlsa[0].columns[2:])
est_sgd_key = list(est_sgd[1])

est_sgd_idx = sorted(range(len(est_sgd_key)), key=est_sgd_key.__getitem__)
est_dlsa_idx = sorted(range(len(est_dlsa_key)), key=est_dlsa_key.__getitem__)

beta_byMLE0 = (est_sgd[0].coef_.T)
if est_sgd[0].fit_intercept:
    beta_byMLE0 = np.concatenate([est_sgd[0].intercept_.reshape(1, 1), beta_byMLE0])
    est_sgd_key.insert(0, 'intercept')

beta_byMLE = beta_byMLE0.copy()
count = 0
for key in est_dlsa_key:
    idx = est_sgd_key.index(key)
    beta_byMLE[count] = beta_byMLE0[idx]
    count += 1

# MLE = pd.DataFrame({'key': np.array(est_sgd_key), 'beta_byMLE': beta_byMLE})


# out = pd.DataFrame({
#     'MLE': beta_byMLE, # SGD
#     'WLSE': est_dlsa[0].iloc[:, 0],
#     'ONESHOT': est_dlsa[0].iloc[:, 1],
#     'DLSA_AIC': est_dlsa[1].iloc[:, 0],
#     'DLSA_BIC': est_dlsa[1].iloc[:, 1]
# })

matrix = np.concatenate([beta_byMLE, # GLOBAL SGD
                         # est_dlsa[0].iloc[:, 0:2], # WLSE, ONE_HOT
                         est_dlsa[1]],
                        axis=1)


out = pd.DataFrame(matrix, index=est_dlsa_key,
                   columns=['MLE',  'DLSA_AIC', 'DLSA_BIC', 'WLSE', 'ONE_SHOT'])
out.to_csv("coef.csv", index_label='Var')
