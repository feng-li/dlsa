#! /usr/bin/env python3

#import findspark
#findspark.init("/usr/lib/spark-current")
#import pyspark
import numpy as np
from scipy.stats import norm
import time
## spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
#sc = pyspark.SparkContext("yarn", "Test App")
print(sc)

data = norm.rvs(size=10000000)

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
