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


matrix = np.concatenate([(est_sgd[0].coef_.T)[est_sgd_idx], # GLOBAL SGD
                         est_dlsa[0].iloc[est_dlsa_idx, 0:2], # WLSE, ONE_HOT
                         est_dlsa[1].iloc[est_dlsa_idx, :]],
                        axis=1)


out = pd.DataFrame(matrix, index=sorted(est_dlsa_key), columns=['MLE', 'WLSE', 'ONE_HOT', 'DLSA_AIC', 'DLSA_BIC'])
out.to_csv("coef.csv", index_label='Var')
