#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark

import os, sys, time
from datetime import timedelta

# from hurry.filesize import size
from math import ceil
import pickle
import numpy as np
import pandas as pd
import string
from math import ceil

from pyspark.sql.types import *
from pyspark.sql import functions
from pyspark.sql.functions import pandas_udf, PandasUDFType,  monotonically_increasing_id

from  dlsa import dlsa, dlsa_r, dlsa_mapred

# os.chdir("dlsa") # TEMP code
from  models import simulate_logistic, logistic_model
from  model_eval import logistic_model_eval_sdf

from  utils import clean_airlinedata, insert_partition_id_pdf
from  utils_spark import convert_schema

from sklearn.linear_model import LogisticRegression

from rpy2.robjects import numpy2ri

spark = pyspark.sql.SparkSession.builder.appName("Spark DLSA App").getOrCreate()

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
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
using_data = "real_hdfs" # ["simulated_pdf", "real_pdf", "real_hdfs"
partition_method = "systematic"
model_saved_file_name = '~/running/logistic_dlsa_model_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.pkl'

# If save data descriptive statistics
# data_info = []
data_info_path = {'save': False, 'path': "~/running/data_raw/data_info.csv"}

# Model settings
#-----------------------------------------------------------------------------------------
fit_intercept = True

# Settings for using simulated data
#-----------------------------------------------------------------------------------------
if using_data in ["simulated_pdf"]:

    n_files = 100 # Sequential loop to avoid Spark OUT_OF_MEM problem
    partition_num_sub = 20
    sample_size_sub = 100000
    sample_size_per_partition = sample_size_sub / partition_num_sub
    p = 200
    Y_name = "label"
    dummy_info = []
    data_info = []
    convert_dummies = []

elif  using_data in ["real_pdf", "real_hdfs"]:
#  Settings for using real data
#-----------------------------------------------------------------------------------------
    # file_path = ['~/running/data_raw/xa' + str(letter) + '.csv.bz2' for letter in string.ascii_lowercase[0:21]] # local file

    file_path = ['/running/data_raw/xa' + str(letter) + '.csv' for letter in string.ascii_lowercase[0:1]] # HDFS file

    # file_path = ['/running/data_raw/allfile.csv'] # HDFS file

    usecols_x = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
                 'CRSArrTime', 'UniqueCarrier', 'ActualElapsedTime', # 'AirTime',
                 'Origin', 'Dest', 'Distance']

    schema_sdf = StructType([
        StructField('Year', IntegerType(), True),
        StructField('Month', IntegerType(), True),
        StructField('DayofMonth', IntegerType(), True),
        StructField('DayOfWeek', IntegerType(), True),
        StructField('DepTime', DoubleType(), True),
        StructField('CRSDepTime', DoubleType(), True),
        StructField('ArrTime', DoubleType(), True),
        StructField('CRSArrTime', DoubleType(), True),
        StructField('UniqueCarrier', StringType(), True),
        StructField('FlightNum', StringType(), True),
        StructField('TailNum', StringType(), True),
        StructField('ActualElapsedTime', DoubleType(), True),
        StructField('CRSElapsedTime',  DoubleType(), True),
        StructField('AirTime',  DoubleType(), True),
        StructField('ArrDelay',  DoubleType(), True),
        StructField('DepDelay',  DoubleType(), True),
        StructField('Origin', StringType(), True),
        StructField('Dest',  StringType(), True),
        StructField('Distance',  DoubleType(), True),
        StructField('TaxiIn',  DoubleType(), True),
        StructField('TaxiOut',  DoubleType(), True),
        StructField('Cancelled',  IntegerType(), True),
        StructField('CancellationCode',  StringType(), True),
        StructField('Diverted',  IntegerType(), True),
        StructField('CarrierDelay', DoubleType(), True),
        StructField('WeatherDelay',  DoubleType(), True),
        StructField('NASDelay',  DoubleType(), True),
        StructField('SecurityDelay',  DoubleType(), True),
        StructField('LateAircraftDelay',  DoubleType(), True)
    ])
    # s = spark.read.schema("col0 INT, col1 DOUBLE")


    dummy_info_path = "~/running/data_raw/dummy_info.pkl"
    dummy_info = pickle.load(open(os.path.expanduser(dummy_info_path), "rb"))
    convert_dummies = list(dummy_info['factor_selected'].keys())

    n_files = len(file_path)
    partition_num_sub = []
    max_sample_size_per_sdf = 100000 # No effect with `real_hdfs` data
    sample_size_per_partition = 100000

    Y_name = "ArrDelay"
    sample_size_sub = []
    memsize_sub = []

# Read or load data chunks into pandas
#-----------------------------------------------------------------------------------------
time_2sdf_sub = []
time_repartition_sub = []

loop_counter = 0
for file_no_i in range(n_files):
    tic_2sdf = time.perf_counter()

    if  using_data == "simulated_pdf":
        if file_no_i == 0:
            # To test performance, we only simulate one subset of data and replicated it.
            data_pdf_i = simulate_logistic(sample_size_sub[0], p,
                                           partition_method,
                                           partition_num_sub)
            memsize_sub0 = sys.getsizeof(data_pdf_i)
        else:
            sample_size_sub.append(sample_size_sub[0])
            memsize_sub.append(memsize_sub0)
            partition_num_sub.append(partition_num_sub[0])

    elif  using_data == "real_pdf": # Read real data
        data_pdf_i0 = clean_airlinedata(os.path.expanduser(file_path[file_no_i]),
                                        fit_intercept=fit_intercept)

        # Create an full-column empty DataFrame and resize current subset
        edf = pd.DataFrame(columns=list(set(dummy_column_names) - set(data_pdf_i0.columns)))
        data_pdf_i = data_pdf_i0.append(edf, sort=True)
        del data_pdf_i0

        data_pdf_i.fillna(0, inplace = True) # Replace append-generated NaN with 0


        partition_num_sub.append(ceil(data_pdf_i.shape[0] / sample_size_per_partition))
        data_pdf_i = insert_partition_id_pdf(data_pdf_i, partition_num_sub[file_no_i],
                                             partition_method)

        sample_size_sub.append(data_pdf_i.shape[0])
        memsize_sub.append(sys.getsizeof(data_pdf_i))


    ## Using HDFS data
    ## ------------------------------
    elif  using_data == "real_hdfs":
        isub = 0 # fixed, never changed

        # Read HDFS to Spark DataFrame
        data_sdf_i = spark.read.csv(file_path[file_no_i], header=True, schema=schema_sdf)
        data_sdf_i = data_sdf_i.select(usecols_x + [Y_name])
        data_sdf_i = data_sdf_i.dropna()

        # Define or transform response variable. Or use
        # https://spark.apache.org/docs/latest/ml-features.html#binarizer
        data_sdf_i = data_sdf_i.withColumn(Y_name,functions.when(data_sdf_i[Y_name] > 0, 1).otherwise(0))

        # Replace dropped factors with `00_OTHERS`. The trick of `00_` prefix will allow
        # user to drop it as the first level when intercept is used.
        for i in dummy_info['factor_dropped'].keys():
            if len(dummy_info['factor_dropped'][i]) > 0:
                data_sdf_i = data_sdf_i.replace(dummy_info['factor_dropped'][i], '00_OTHERS', i)

        sample_size_sub.append(data_sdf_i.count())
        partition_num_sub.append(ceil(sample_size_sub[file_no_i] / sample_size_per_partition))

        ## Add partition ID
        data_sdf_i = data_sdf_i.withColumn(
            "partition_id",
            monotonically_increasing_id() % partition_num_sub[file_no_i])

        ## Create dummy variables We could do it either directly with
        ## https://stackoverflow.com/questions/35879372/pyspark-matrix-with-dummy-variables
        ## or we do it within grouped dlsa (default)

##----------------------------------------------------------------------------------------
## MODEL FITTING ON PARTITIONED DATA
##----------------------------------------------------------------------------------------
    # Split the process into small subs if reading a real big DataFrame which my cause
    # MemoryError
    if  using_data in ["real_pdf", "simulated_pdf"]:
        nsub = ceil(sample_size_sub[file_no_i] / max_sample_size_per_sdf)

        for isub in range(nsub):

            # Convert Pandas DataFrame to Spark DataFrame
            idx_curr_sub = [round(sample_size_sub[file_no_i] / nsub * isub),
                            round(sample_size_sub[file_no_i] / nsub * (isub + 1))]

            data_sdf_isub = spark.createDataFrame(data_pdf_i.iloc[idx_curr_sub[0]:idx_curr_sub[1], ])

            # Union all sequential feeded pdf to sdf.
            if isub == 0:
                data_sdf_i = data_sdf_isub
                # memsize_sub = sys.getsizeof(data_pdf_i)
            else:
                data_sdf_i = data_sdf_i.unionAll(data_sdf_isub)

            loop_counter += 1
            time_elapsed = time.perf_counter() - tic_2sdf
            time_to_go = timedelta(seconds=time_elapsed / loop_counter * (n_files * nsub - loop_counter))
            print('Creating Spark DataFrame:\t' + str(isub) + '/' + str(nsub))
            print('Time elapsed:\t' + str(timedelta(seconds=time.perf_counter() - tic_2sdf))
                  + '.\tTime to go:\t' + str(time_to_go))


    time_2sdf_sub.append(time.perf_counter() - tic_2sdf)


##----------------------------------------------------------------------------------------
## MODELING ON PARTITIONED DATA
##----------------------------------------------------------------------------------------
    # Descriptive statistics
    if len(data_info_path) > 0:
        if data_info_path["save"] == True:
            data_info = data_sdf_i.describe().toPandas() # descriptive statistics
            data_info.to_csv(os.path.expanduser(data_info_path["path"]), index=False)
            print("Descriptive statistics for data are saved to:\t" + data_info_path["path"])
        else:
            data_info = pd.read_csv(os.path.expanduser(data_info_path["path"]))
            print("Descriptive statistics for data are loaded from file:\t" + data_info_path["path"])


    # Standardized the data. This requires 'requirement failed: Column Year must be of
    # type org.apache.spark.ml.linalg.VectorUD. We do it with Map model.

    # from pyspark.ml.feature import StandardScaler
    # scaler = StandardScaler(inputCol="Distance", outputCol="scaledDistance",
    #                         withStd=True, withMean=True)
    # scalerModel = scaler.fit(data_sdf_i)
    # scaledData = scalerModel.transform(data_sdf_i)

    tic_repartition = time.perf_counter()
    data_sdf_i = data_sdf_i.repartition(partition_num_sub[file_no_i], "partition_id")
    time_repartition_sub.append(time.perf_counter() - tic_repartition)

    ## Register a user defined function via the Pandas UDF
    schema_beta = StructType(
        [StructField('par_id', IntegerType(), True),
         StructField('coef', DoubleType(), True),
         StructField('Sig_invMcoef', DoubleType(), True)]
        + convert_schema(usecols_x, dummy_info, fit_intercept))

    @pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
    def logistic_model_udf(sample_df):
        return logistic_model(sample_df=sample_df,
                              Y_name=Y_name,
                              fit_intercept=fit_intercept,
                              dummy_info=dummy_info,
                              data_info=data_info)

    # pdb.set_trace()
    # partition the data and run the UDF
    model_mapped_sdf_i = data_sdf_i.groupby("partition_id").apply(logistic_model_udf)

    # Union all sequential mapped results.
    if file_no_i == 0 & isub == 0:
        model_mapped_sdf = model_mapped_sdf_i
        # memsize_sub = sys.getsizeof(data_pdf_i)
    else:
        model_mapped_sdf = model_mapped_sdf.unionAll(model_mapped_sdf_i)

##----------------------------------------------------------------------------------------
## AGGREGATING THE MODEL ESTIMATES
##----------------------------------------------------------------------------------------
if using_data == "simulated_pdf":
    p = data_pdf_i.shape[1]

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
                fit_intercept=fit_intercept)

time_dlsa = time.perf_counter() - tic_dlsa
##----------------------------------------------------------------------------------------
## Model Evaluation
##----------------------------------------------------------------------------------------
tic_model_eval = time.perf_counter()

out_par = out_dlsa
out_par["par_byOLS"] = Sig_inv_beta["par_byOLS"]
out_par["par_byONESHOT"] = Sig_inv_beta["par_byONESHOT"]

out_model_eval = logistic_model_eval_sdf(
    data_sdf=data_sdf_i,
    par=out_par,
    fit_intercept=fit_intercept,
    Y_name=Y_name,
    dummy_info=dummy_info,
    data_info=data_info)

time_model_eval = time.perf_counter() - tic_model_eval
##----------------------------------------------------------------------------------------
## PRINT OUTPUT
##----------------------------------------------------------------------------------------
memsize_total = sum(memsize_sub)
partition_num = sum(partition_num_sub)
time_repartition = sum(time_repartition_sub)
# time_2sdf = sum(time_2sdf_sub)
# sample_size_per_partition = sample_size / partition_num

out_time = pd.DataFrame({
    "sample_size": sample_size,
    "sample_size_per_partition": sample_size_per_partition,
    "n_par": len(schema_beta) - 3,
    "partition_num": partition_num,
    "memsize_total": memsize_total,
    # "time_2sdf": time_2sdf,
    "time_repartition": time_repartition,
    "time_mapred": time_mapred,
    "time_dlsa": time_dlsa,
    "time_model_eval": time_model_eval}, index=[0])

# save the model to pickle, use pd.read_pickle("test.pkl") to load it.
# out_dlas.to_pickle("test.pkl")
out = [Sig_inv_beta, out_dlsa, out_par, out_model_eval, out_time]
pickle.dump(out, open(os.path.expanduser(model_saved_file_name), 'wb'))
print("Model results are saved to:\t" + model_saved_file_name)

# print(", ".join(format(x, "10.2f") for x in out_time))
print("\nModel Summary:\n")
print(out_time.to_string(index=False))

print("\nModel Evaluation:\n")
print(out_model_eval.to_string(index=False))

print("\nDLSA Coefficients:\n")
print(out_par.to_string())

# Verify with Pure R implementation.
# numpy2ri.activate()
# out_dlsa_r = dlsa_r(Sig_inv_=np.asarray(Sig_inv_beta.iloc[:, 2:]),
#                     beta_=np.asarray(Sig_inv_beta["par_byOLS"]),
#                     sample_size=data_sdf.count(), intercept=False)
# numpy2ri.deactivate()

# out_dlsa = dlsa(Sig_inv_=Sig_inv,
#                 beta_=par_byOLS,
#                 sample_size=data_sdf.count(), intercept=False)
