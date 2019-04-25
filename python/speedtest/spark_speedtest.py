#! /usr/bin/env python3

# import findspark
# findspark.init("/usr/lib/spark-current")
import pyspark
import numpy as np
from scipy.stats import norm
import time
import sys

## spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
# sc = pyspark.SparkContext("yarn", "Speed Test App")


## PKU cluster
spark_master = os.getenv('SPARK_MASTER')
spark_master = 'spark://' + spark_master + ':7077'
conf = pyspark.SparkConf()
conf.setMaster(spark_master)
conf.setAppName('spark-test')
sc = pyspark.SparkContext(conf=conf)

datasize = [2 ** x for x in range(25)]

print(", ".join(["len", "memsize", "time_parallelize",  "time_broadcast", "time_clusterrun", "time_singlerun"]))

for i in datasize:

    data = norm.rvs(size=i)
    memsize = sys.getsizeof(data)

    tic = time.clock()
    rdd = sc.parallelize(data)
    toc = time.clock()
    time_parallelize = toc - tic

    ## Cluster time
    tic = time.clock()
    out = rdd.mean()
    toc = time.clock()
    time_clusterrun = toc - tic

    tic = time.clock()
    rdd1 = sc.broadcast(data)
    toc = time.clock()
    time_broadcast = toc - tic

    ## Single Machine
    tic = time.clock()
    out = data.mean()
    toc = time.clock()
    time_singlerun = toc - tic

    out = [i, memsize, time_parallelize, time_broadcast, time_clusterrun, time_singlerun]
    print(", ".join(format(x, "10.4f") for x in out))

sc.stop()
