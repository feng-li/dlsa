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
# Basic settings
nsub = 100 # Sequential loop to avoid Spark OUT_OF_MEM problem
sample_size_sub = 100000
p = 500
partition_num_sub = 10
partition_method = "systematic"

for isub in range(nsub):

    # Read or load data chunks into pandas

    if isub == 0: # To test performance, we only simulate one subset of data and replicated it.
        data_pdf_i = simulate_logistic(sample_size_sub, p,
                                       partition_method,
                                       partition_num_sub)

    # Convert Pandas DataFrame to Spark DataFrame
    data_sdf_i = spark.createDataFrame(data_pdf_i)

    # Repartition

    tic_repartition = time.perf_counter()
    data_sdf_i = data_sdf_i.repartition(partition_num_sub, "partition_id")
    time_repartition_sub = time.perf_counter() - tic_repartition

    ##----------------------------------------------------------------------------------------
    ## PARTITIONED LOGISTIC REGRESSION
    ##----------------------------------------------------------------------------------------
    fit_intercept = False

    # Register a user defined function via the Pandas UDF
    schema_beta = StructType(
        [StructField('par_id', IntegerType(), True),
         StructField('coef', DoubleType(), True),
         StructField('Sig_invMcoef', DoubleType(), True)]
        + data_sdf_i.schema.fields[2:])

    @pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
    def logistic_model_udf(sample_df):
        return logistic_model(sample_df=sample_df, fit_intercept=fit_intercept)

    # pdb.set_trace()
    # partition the data and run the UDF
    model_mapped_sdf_i = data_sdf_i.groupby("partition_id").apply(logistic_model_udf)

    # Union all sequential mapped results.
    if isub == 0:
        model_mapped_sdf = model_mapped_sdf_i
        memsize_sub = sys.getsizeof(data_pdf_i)
    else:
        model_mapped_sdf = model_mapped_sdf.unionAll(model_mapped_sdf_i)

##----------------------------------------------------------------------------------------
## FINAL OUTPUT
##----------------------------------------------------------------------------------------
# sample_size=model_mapped_sdf.count()
sample_size = sample_size_sub * nsub

# Obtain Sig_inv and beta
tic_mapred = time.perf_counter()
Sig_inv_beta = dlsa_mapred(model_mapped_sdf)
time_mapred = time.perf_counter() - tic_mapred

tic_dlsa = time.perf_counter()
out_dlsa = dlsa(Sig_inv_=Sig_inv_beta.iloc[:, 2:],
                beta_=Sig_inv_beta["par_byOLS"],
                sample_size=sample_size,
                intercept=fit_intercept)

time_dlsa = time.perf_counter() - tic_dlsa

##----------------------------------------------------------------------------------------
## PRINT OUTPUT
##----------------------------------------------------------------------------------------
memsize_total = memsize_sub * nsub
partition_num = partition_num_sub * nsub
time_repartition = time_repartition_sub * nsub
sample_size_per_partition = sample_size / partition_num


out_time = pd.DataFrame(
    {"sample_size": sample_size,
     "sample_size_sub": sample_size_sub,
     "sample_size_per_partition": sample_size_per_partition,
     "p": p,
     "partition_num": partition_num,
     "memsize_total": memsize_total,
     "time_repartition": time_repartition,
     "time_mapred": time_mapred,
     "time_dlsa": time_dlsa}, index=[0])

# print(", ".join(format(x, "10.2f") for x in out_time))
print("Model Summary:\n")
print(out_time.to_string())
print("\nDLSA Coefficients:\n")
print(out_dlsa.to_string())

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
