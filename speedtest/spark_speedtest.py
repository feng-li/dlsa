#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
from pyspark import SparkContext, SparkConf
import numpy as np
from scipy.stats import norm
import time
## spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
sc = pyspark.SparkContext("local", "Test App")

data = norm.rvs(size=100000000)

r = sc.parallelize(data)
tic = time.clock()
out = r.mean()
toc = time.clock()
print(toc - tic)


## Single Machine
tic = time.clock()
out = data.mean()
toc = time.clock()
print(toc - tic)
