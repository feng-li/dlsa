#! /usr/bin/python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
import pandas as pd
import time
import sys
import pyarrow as pa
from pyspark.mllib.regression import LabeledPoint

tic0 = time.clock()
sc = pyspark.SparkContext("yarn", "Speed Test App")

## Simulate Data
n = 10
p = 3
p1 = int(p * 0.3)
# hdfs_path = "/data/simulated/logistic.txt"

## Connect to a default HDFS system
fs = pa.hdfs.connect()

features = np.random.rand(n, p) - 0.5

beta = np.zeros(p).reshape(p, 1)
beta[:p1] = 1
prob = 1 / (1 + np.exp(-features.dot(beta)))

label = np.zeros(n).reshape(n, 1)
for i in range(n):
    # TODO: REMOVE loop
    label[i] = np.random.binomial(n=1,p=prob[i], size=1)
df = np.concatenate((label, features), 1)
pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])

dff = map(lambda x: LabeledPoint(int(x[0]), x[1:]), df)
parsedData = sc.parallelize(dff, 12)

sc.parallelize(dff, 12).saveAsTextFile("/data/simulation/")

loadrdd = pyspark.SparkContext.wholeTextFiles("/data/simulation")
# table = pa.Table.from_pandas(pdf)
# with fs.open(hdfs_path, "wb") as f:
#     f.write(table)





# ## TRUE beta
# beta = np.zeros(p).reshape(p, 1)
# beta[:p1] = 1
# prob = 1 / (1 + np.exp(-features.dot(beta)))

# label = np.zeros(n).reshape(n, 1)
# for i in range(n):
#     # TODO: REMOVE loop
#     label[i] = np.random.binomial(n=1,p=prob[i], size=1)

# df = np.concatenate((label, features), 1)
# # pdf = pd.DataFrame(df, columns=["label"] + ["x" + str(x) for x in range(p)])
# memsize = sys.getsizeof(df)
