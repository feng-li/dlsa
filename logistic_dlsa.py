#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark

import os, sys, time

# from hurry.filesize import size

import numpy as np
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

from dlsa import dlsa, dlsa_r, dlsa_mapred
# os.chdir("dlsa") # TEMP code
from models import simulate_logistic, logistic_model
from sklearn.linear_model import LogisticRegression

from rpy2.robjects import numpy2ri

spark = pyspark.sql.SparkSession.builder.appName("Spark DLSA App").getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")


# FIXME: PATH BUG
spark.sparkContext.addPyFile("/home/lifeng/code/dlsa/models.py")

# BASH compatible
# spark.sparkContext.addPyFile(os.path.dirname(os.path.abspath(__file__)) + "/models.py")

# Python compatible
# spark.sparkContext.addPyFile(os.path.dirname(os.path.abspath(__file__)) + "/dlsa/models.py")

# https://docs.azuredatabricks.net/spark/latest/spark-sql/udf-python-pandas.html#setting-arrow-batch-size
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10000) # default

# spark.conf.set("spark.sql.shuffle.partitions", 10)
# print(spark.conf.get("spark.sql.shuffle.partitions"))

##----------------------------------------------------------------------------------------
## USING REAL DATA
##----------------------------------------------------------------------------------------
# load the CSV as a Spark data frame
# data_df = pd.read_csv("../data/games-expand.csv")
# data_sdf = spark.createDataFrame(pandas_df)

# FIXME: Real data should add an arbitrary partition id.

# assign a row ID and a partition ID using Spark SQL
# FIXME: WARN WindowExec: No Partition Defined for Window operation! Moving all data to a
# single partition, this can cause serious performance
# degradation. https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
# data_sdf.createOrReplaceTempView("data_sdf")
# data_sdf = spark.sql("""
# select *, row_id%20 as partition_id
# from (
#   select *, row_number() over (order by rand()) as row_id
#   from data_sdf
# )
# """)

##----------------------------------------------------------------------------------------
## USING SIMULATED DATA
##----------------------------------------------------------------------------------------
sample_size=10000
p=200
partition_method="systematic"
partition_num=10

nsub = 5

data_pdf = simulate_logistic(sample_size, p,
                             partition_method,
                             partition_num)
memsize0 = sys.getsizeof(data_pdf)

data_sdf = spark.createDataFrame(data_pdf)


# Do a loop to union sdf
for isub in range(nsub - 1):

    data_pdfi = simulate_logistic(sample_size, p,
                                  partition_method,
                                  partition_num)
    data_sdfi = spark.createDataFrame(data_pdfi)

    data_sdf = data_sdf.unionAll(data_sdfi)


memsize_total = memsize0 * nsub

# Repartition

tic_repartition = time.clock()
data_sdf = data_sdf.repartition(partition_num, "partition_id")
time_repartition = time.clock() - tic_repartition
##----------------------------------------------------------------------------------------
## LOGISTIC REGRESSION WITH DLSA
##----------------------------------------------------------------------------------------
fit_intercept = False

# Register a user defined function via the Pandas UDF
schema_beta = StructType(
    [StructField('par_id', IntegerType(), True),
     StructField('coef', DoubleType(), True),
     StructField('Sig_invMcoef', DoubleType(), True)]
    + data_sdf.schema.fields[2:])

@pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
def logistic_model_udf(sample_df):
    return logistic_model(sample_df=sample_df, fit_intercept=fit_intercept)

tic_mapred = time.clock()
Sig_inv_beta = dlsa_mapred(logistic_model_udf, data_sdf, "partition_id")
time_mapred = time.clock() - tic_mapred
##----------------------------------------------------------------------------------------
## FINAL OUTPUT
##----------------------------------------------------------------------------------------
tic_dlsa = time.clock()
out_dlsa = dlsa(Sig_inv_=Sig_inv_beta.iloc[:, 2:],
                beta_=Sig_inv_beta["par_byOLS"],
                sample_size=data_sdf.count(), intercept=False)

time_dlsa = time.clock() - tic_dlsa

##----------------------------------------------------------------------------------------
## PRINT OUTPUT
##----------------------------------------------------------------------------------------

out_time = [sample_size * nsub, p, partition_num, memsize_total, time_repartition, time_mapred, time_dlsa]
# print("Model Summary:")
print(", ".join(format(x, "10.2f") for x in out_time))

print(out_dlsa)

# save the model to pickle, use pd.read_pickle("test.pkl") to load it.
# out_dlas.to_pickle("test.pkl")


# Verify with Pure R implementation.
# numpy2ri.activate()
# out_dlsa_r = dlsa_r(Sig_inv_=np.asarray(Sig_inv_beta.iloc[:, 2:]),
#                     beta_=np.asarray(Sig_inv_beta["par_byOLS"]),
#                     sample_size=data_sdf.count(), intercept=False)
# numpy2ri.deactivate()

# out_dlsa = dlsa(Sig_inv_=Sig_inv,
#                 beta_=par_byOLS,
#                 sample_size=data_sdf.count(), intercept=False)
