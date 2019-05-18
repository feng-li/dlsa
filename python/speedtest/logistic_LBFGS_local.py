#! /usr/bin/python3

import numpy as np
import pandas as pd
import time
import sys
from sklearn.linear_model import LogisticRegression

tic0 = time.clock()

## Simulate Data
n = 5000000
p = 500
p1 = int(p * 0.3)

features = np.random.rand(n, p) - 0.5

## TRUE beta
beta = np.zeros(p).reshape(p, 1)
beta[:p1] = 1
prob = 1 / (1 + np.exp(-features.dot(beta)))

label = np.zeros(n).reshape(n, 1)
for i in range(n):
    # TODO: REMOVE loop
    label[i] = np.random.binomial(n=1,p=prob[i], size=1)

df = np.concatenate((label, features), 1)
# pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
memsize = sys.getsizeof(df)


tic = time.clock()
time_parallelize = time.clock() - tic

##----------------------------------------------------------------------------------------
## Logistic Regression with LBFGS
##----------------------------------------------------------------------------------------

tic = time.clock()
# Model configuration
# lr = LogisticRegression(penalty='l1', fit_intercept=False, solver="saga").fit(features, label)
time_clusterrun = time.clock() - tic

time_wallclock = time.clock() - tic0

out = [n, p, memsize, time_parallelize, time_clusterrun, time_wallclock]
print(", ".join(format(x, "10.4f") for x in out))
