#! /usr/bin/env python3

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

from dlsa import simulate_logistic

spark = pyspark.sql.SparkSession.builder.appName(
    "Spark Native Logistic Regression App").getOrCreate()

tic0 = time.clock()
##----------------------------------------------------------------------------------------
## Logistic Regression with SGD
##----------------------------------------------------------------------------------------
sample_size = 5000
p = 50
partition_method = "systematic"
partition_num = 20

data_pdf = simulate_logistic(sample_size, p, partition_method, partition_num)
data_sdf = spark.createDataFrame(data_pdf)

memsize = sys.getsizeof(data_pdf)

assembler = VectorAssembler(inputCols=["x" + str(x) for x in range(p)],
                            outputCol="features")

tic = time.clock()
parsedData = assembler.transform(data_sdf)
time_parallelize = time.clock() - tic

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

out = [
    sample_size, p, memsize, time_parallelize, time_clusterrun, time_wallclock
]
print(", ".join(format(x, "10.4f") for x in out))
