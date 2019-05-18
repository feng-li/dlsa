#! /usr/bin/python3

import pyspark
sc = pyspark.SparkContext (
   "local",
   "storagelevel app"
)
rdd1 = sc.parallelize([1,2])
rdd1.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
rdd1.getStorageLevel()
print(rdd1.getStorageLevel())
