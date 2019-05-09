#! /usr/bin/python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
import pandas as pd
import time
import sys

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

tic0 = time.clock()
spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

## Simulate Data
n = 500000
p = 500

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
pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
data = spark.createDataFrame(pdf)

assembler = VectorAssembler(
    inputCols=["x" + str(x) for x in range(p)],
    outputCol="features")

tic = time.clock()
parsedData = assembler.transform(data)
time_parallelize = time.clock() - tic

##----------------------------------------------------------------------------------------
## Logistic Regression with SGD
##----------------------------------------------------------------------------------------

tic = time.clock()
# Model configuration
lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(parsedData)
time_clusterrun = time.clock() - tic

# Model fitted
print(lrModel.intercept)
print(lrModel.coefficients)

time_wallclock = time.clock() - tic0

out = [n, p, memsize, time_parallelize, time_clusterrun, time_wallclock]
print(", ".join(format(x, "10.4f") for x in out))

spark.stop()
