#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set("spark.sql.shuffle.partitions", 10)
print(spark.conf.get("spark.sql.shuffle.partitions"))


##----------------------------------------------------------------------------------------
## USING REAL DATA
##----------------------------------------------------------------------------------------
# load the CSV as a Spark data frame
# data_df = pd.read_csv("../data/games-expand.csv")
# data_sdf = spark.createDataFrame(pandas_df)

##----------------------------------------------------------------------------------------
## USING SIMULATED DATA
##----------------------------------------------------------------------------------------

## Simulate Data
n = 50000
p = 50
p1 = int(p * 0.3)

## TRUE beta
beta = np.zeros(p).reshape(p, 1)
beta[:p1] = 1

## Simulate features
features = np.random.rand(n, p) - 0.5
prob = 1 / (1 + np.exp(-features.dot(beta)))

## Simulate label
label = np.zeros(n).reshape(n, 1)
for i in range(n):
    # TODO: REMOVE loop
    label[i] = np.random.binomial(n=1,p=prob[i], size=1)

data_np = np.concatenate((label, features), 1)
data_pdf = pd.DataFrame(data_np, columns=["label"] + ["x" + str(x) for x in range(p)])
data_sdf = spark.createDataFrame(data_pdf)

# define a beta schema
schema_beta = data_sdf.schema[1:]

##----------------------------------------------------------------------------------------
## Logistic Regression with SGD
##----------------------------------------------------------------------------------------
# assembler = VectorAssembler(
#     inputCols=["x" + str(x) for x in range(p)],
#     outputCol="features")

# tic = time.clock()
# parsedData = assembler.transform(data_sdf)
# time_parallelize = time.clock() - tic

# tic = time.clock()
# # Model configuration
# lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

# # Fit the model
# lrModel = lr.fit(parsedData)
# time_clusterrun = time.clock() - tic

# # Model fitted
# print(lrModel.intercept)
# print(lrModel.coefficients)

# time_wallclock = time.clock() - tic0

# out = [n, p, memsize, time_parallelize, time_clusterrun, time_wallclock]
# print(", ".join(format(x, "10.4f") for x in out))

##----------------------------------------------------------------------------------------
## LOGISTIC REGRESSION WITH DLSA
##----------------------------------------------------------------------------------------

# assign a user ID and a partition ID using Spark SQL
# FIXME: WARN WindowExec: No Partition Defined for Window operation! Moving all data to a
# single partition, this can cause serious performance
# degradation. https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
data_sdf.createOrReplaceTempView("data_sdf")
data_sdf = spark.sql("""
select *, row_id%20 as partition_id
from (
  select *, row_number() over (order by rand()) as row_id
  from data_sdf
)
""")

##----------------------------------------------------------------------------------------
## APPLY USER-DEFINED FUNCTIONS TO PARTITIONED DATA
##----------------------------------------------------------------------------------------

# define the Pandas UDF
@pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
def logistic_model(sample_df):
    # run the model on the partitioned data set
    x_train = sample_df.drop(['label', 'row_id', 'partition_id'], axis=1)
    y_train = sample_df["label"]
    model = LogisticRegression(solver="lbfgs", fit_intercept=False)
    model.fit(x_train, y_train)
    coef = model.coef_

    return pd.DataFrame(coef)

# partition the data and run the UDF
results = data_sdf.groupby('partition_id').apply(logistic_model)


##----------------------------------------------------------------------------------------
## MERGE AND DEBIAS
##----------------------------------------------------------------------------------------

print(results.toPandas())
