#! /usr/bin/python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
import pandas as pd
import time
import sys

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

tic0 = time.clock()
sc = pyspark.SparkContext("yarn", "Speed Test App")


## Simulation settings
save_to_hdfs = False

nsub = 5
n = 100000
p = 500
p1 = int(p * 0.3)

beta = np.zeros(p).reshape(p, 1)
beta[:p1] = 1

time_parallelize = 0
for isub in range(nsub):

    features = np.random.rand(n, p) - 0.5

    ## TRUE beta
    prob = 1 / (1 + np.exp(-features.dot(beta)))

    label = np.zeros(n).reshape(n, 1)
    for i in range(n):
        # TODO: REMOVE loop
        label[i] = np.random.binomial(n=1,p=prob[i], size=1)

    df = np.concatenate((label, features), 1)
    # pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
    memsize_isub = sys.getsizeof(df)

    dff = map(lambda x: LabeledPoint(int(x[0]), x[1:]), df)

    tic = time.clock()
    parsedDataNew = sc.parallelize(dff)

    if isub == 0:
        parsedData = parsedDataNew
    else:
        parsedData = parsedData.union(parsedDataNew)


    ## https://spark.apache.org/docs/latest/rdd-programming-guide.html#rdd-persistence
    ## https://spark.apache.org/docs/latest/api/python/pyspark.html#pyspark.StorageLevel
    ## parsedDataNew.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    parsedData.persist(pyspark.StorageLevel.DISK_ONLY)

    parsedDataNew.unpersist() # delete rdd


    time_parallelize += (time.clock() - tic)


del dff, df, parsedDataNew

if save_to_hdfs == True:
    parsedData.saveAsTextFile("/data/simulation/")

##----------------------------------------------------------------------------------------
## Logistic Regression with LBFGS
##----------------------------------------------------------------------------------------

tic = time.clock()
# Model configuration
lr = LogisticRegressionWithLBFGS.train(parsedData, iterations=10000)
time_clusterrun = time.clock() - tic

time_wallclock = time.clock() - tic0

memsize = memsize_isub * (nsub + 1)

out = [n * nsub, p, memsize, time_parallelize, time_clusterrun, time_wallclock]
print(", ".join(format(x, "10.4f") for x in out))

sc.stop()
