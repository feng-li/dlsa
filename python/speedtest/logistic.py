#! /usr/bin/python3


import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import sys
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
## sc = pyspark.SparkContext("yarn", "Speed Test App")

n = 100000
p = 200
features = np.random.rand(n, p)
label = np.random.binomial(n=1,p=0.6, size=n).reshape(n, 1)
df = np.concatenate((label, features), 1)
pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
data = spark.createDataFrame(pdf)

assembler = VectorAssembler(
    inputCols=["x" + str(x) for x in range(p)],
    outputCol="features")

output = assembler.transform(data)

tic = time.clock()
# Model configuration
lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(output)

# Model fitted
# print(lrModel.intercept)
# print(lrModel.coefficients)
toc = time.clock()

print(toc - tic)
