#! /usr/bin/python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import sys

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
sc = pyspark.SparkContext("yarn", "Speed Test App")


n = 10000
p = 300
features = np.random.rand(n, p)
label = np.random.binomial(n=1,p=0.6, size=n).reshape(n, 1)
df = np.concatenate((label, features), 1)
# pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
dff = map(lambda x: LabeledPoint(int(x[0]), x[1:]), df)
parsedData = sc.parallelize(dff)

##----------------------------------------------------------------------------------------
## Logistic Regression with LBFGS
##----------------------------------------------------------------------------------------

tic = time.clock()
# Model configuration
lr = LogisticRegressionWithLBFGS.train(parsedData, iterations=10000)
toc = time.clock()

print(toc - tic)
sc.stop()
