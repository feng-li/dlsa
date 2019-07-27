#! /usr/bin/env python3

import pickle
import os
import pandas as pd
import numpy as np

sgd = "~/running/logistic_sgd_model_2019-07-27-00:08:04.pkl"
dlsa = "~/running/logistic_dlsa_model_2019-07-26-23:54:15.pkl"

est_sgd = pickle.load(open(os.path.expanduser(sgd), "rb"))
est_dlsa = pd.read_pickle(os.path.expanduser(dlsa))

matrix = np.concatenate([est_sgd.coef_.T, est_dlsa[1].iloc[1:, :], est_dlsa[0].iloc[1:, 0:2]], axis=1)
np.savetxt("coef.csv", matrix, delimiter=",")
