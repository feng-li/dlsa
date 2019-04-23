#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
from scipy.stats import norm
import time
import sys

## spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
sc = pyspark.SparkContext("yarn", "Speed Test App")

datasize = [2 ** x for x in range(20)]

print(", ".join(["len", "memsize", "time_comm", "time_clusterrun", "time_singlerun"]))

for i in datasize:

    data = norm.rvs(size=i)
    memsize = sys.getsizeof(data)

    tic = time.clock()
    rdd = sc.parallelize(data)
    toc = time.clock()
    time_comm = toc - tic

    ## Cluster time
    tic = time.clock()
    out = rdd.mean()
    toc = time.clock()
    time_cluster = toc - tic

    ## Single Machine
    tic = time.clock()
    out = data.mean()
    toc = time.clock()
    time_single = toc - tic

    out = [i, memsize, time_comm, time_cluster, time_single]
    print(", ".join(format(x, "10.8f") for x in out))
    ## print(i, memsize, time_trans, time_cluster, time_single)

sc.stop()
