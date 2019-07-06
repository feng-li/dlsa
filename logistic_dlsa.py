#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark

import os, sys, time

# from hurry.filesize import size
from math import ceil

import numpy as np
import pandas as pd

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

from dlsa import dlsa, dlsa_r, dlsa_mapred

# os.chdir("dlsa") # TEMP code
from models import simulate_logistic, logistic_model
from utils import clean_airlinedata, insert_partition_id_pdf, convert_schema

from sklearn.linear_model import LogisticRegression

from rpy2.robjects import numpy2ri

spark = pyspark.sql.SparkSession.builder.appName("Spark DLSA App").getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.enabled", "false")
spark.conf.set("spark.sql.execution.arrow.fallback.enabled", "true")


# FIXME: PATH BUG
spark.sparkContext.addPyFile("/home/lifeng/code/dlsa/models.py")
spark.sparkContext.addPyFile("/home/lifeng/code/dlsa/utils.py")

# BASH compatible
# spark.sparkContext.addPyFile(os.path.dirname(os.path.abspath(__file__)) + "/models.py")

# Python compatible
# spark.sparkContext.addPyFile(os.path.dirname(os.path.abspath(__file__)) + "/dlsa/models.py")

# https://docs.azuredatabricks.net/spark/latest/spark-sql/udf-python-pandas.html#setting-arrow-batch-size
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10000) # default

# spark.conf.set("spark.sql.shuffle.partitions", 10)
# print(spark.conf.get("spark.sql.shuffle.partitions"))

##----------------------------------------------------------------------------------------
## SETTINGS
##----------------------------------------------------------------------------------------

# General  settings
#-----------------------------------------------------------------------------------------
using_simulated_data = False
partition_method = "systematic"

# Settings for using simulated data
#-----------------------------------------------------------------------------------------
if using_simulated_data:

    nsub = 100 # Sequential loop to avoid Spark OUT_OF_MEM problem
    partition_num_sub = 20
    sample_size_sub = 100000
    sample_size_per_partition = sample_size_sub / partition_num_sub
    p = 200
    Y_name = "label"

else:
#  Settings for using real data
#-----------------------------------------------------------------------------------------
    # file_path = ['~/running/data/' + str(year) + '.csv.bz2' for year in range(1987, 2008 + 1)]
    file_path = ['~/running/data/' + str(year) + '.csv.bz2' for year in range(1987, 1987 + 1)]

    nsub = len(file_path)
    partition_num_sub = []
    sample_size_per_partition = 50000

    Y_name = "ArrDelay"
    sample_size_sub = []
    memsize_sub = []

# Model settings
#-----------------------------------------------------------------------------------------
fit_intercept = False

# Read or load data chunks into pandas
#-----------------------------------------------------------------------------------------
for isub in range(nsub):
    if using_simulated_data:
        if isub == 0:
            # To test performance, we only simulate one subset of data and replicated it.
            data_pdf_i = simulate_logistic(sample_size_sub[0], p,
                                           partition_method,
                                           partition_num_sub)
            memsize_sub0 = sys.getsizeof(data_pdf_i)
        else:
            sample_size_sub.append(sample_size_sub[0])
            memsize_sub.append(memsize_sub0)
            partition_num_sub.append(partition_num_sub[0])

    else: # Read real data
        data_pdf_i0 = clean_airlinedata(os.path.expanduser(file_path[isub]))
        partition_num_sub = [1]
        # partition_num_sub.append(ceil(data_pdf_i0.shape[0] / sample_size_per_partition))
        data_pdf_i = insert_partition_id_pdf(data_pdf_i0, partition_num_sub[isub],
                                             partition_method)

        sample_size_sub.append(data_pdf_i.shape[0])
        memsize_sub.append(sys.getsizeof(data_pdf_i))
##----------------------------------------------------------------------------------------
## MODEL FITTING ON PARTITIONED DATA
##----------------------------------------------------------------------------------------
    # Convert Pandas DataFrame to Spark DataFrame
    data_sdf_i = spark.createDataFrame(data_pdf_i)

    # Repartition
    if isub == 0:
        time_repartition_sub = []

    tic_repartition = time.perf_counter()
    data_sdf_i = data_sdf_i.repartition(partition_num_sub[isub], "partition_id")

    time_repartition_sub.append(time.perf_counter() - tic_repartition)

    ##----------------------------------------------------------------------------------------
    ## PARTITIONED LOGISTIC REGRESSION
    ##----------------------------------------------------------------------------------------

    # Register a user defined function via the Pandas UDF
    schema_beta = StructType(
        [StructField('par_id', IntegerType(), True),
         StructField('coef', DoubleType(), True),
         StructField('Sig_invMcoef', DoubleType(), True)]
        + convert_schema(data_sdf_i.schema.fields[2:],  out_type=DoubleType()))

    @pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
    def logistic_model_udf(sample_df):
        return logistic_model(sample_df=sample_df, Y_name=Y_name, fit_intercept=fit_intercept)

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
## AGGREGATING THE MODEL ESTIMATES
##----------------------------------------------------------------------------------------
if using_simulated_data == False:
    p = data_pdf_i0.shape[1]

# sample_size=model_mapped_sdf.count()
sample_size = sum(sample_size_sub)

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
memsize_total = sum(memsize_sub)
partition_num = sum(partition_num_sub)
time_repartition = sum(time_repartition_sub)
# sample_size_per_partition = sample_size / partition_num

out_time = pd.DataFrame(
    {"sample_size": sample_size,
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
