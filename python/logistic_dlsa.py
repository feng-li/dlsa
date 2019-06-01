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
# spark.conf.set("spark.sql.shuffle.partitions", 10)
print(spark.conf.get("spark.sql.shuffle.partitions"))


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

## Simulate Data
n = 5000
p = 50
p1 = int(p * 0.3)

partition_method = "systematic"
partition_num = 4

## TRUE beta
beta = np.zeros(p).reshape(p, 1)
beta[:p1] = 1

## Simulate features
features = np.random.rand(n, p) - 0.5
prob = 1 / (1 + np.exp(-features.dot(beta)))

## Simulate label
label = np.zeros(n).reshape(n, 1)
partition_id = np.zeros(n).reshape(n, 1)
for i in range(n):
    # TODO: REMOVE loop
    label[i] = np.random.binomial(n=1,p=prob[i], size=1)

    if partition_method == "systematic":
        partition_id[i] = i % partition_num
    else:
        raise Exception("No such partition method implemented!")


data_np = np.concatenate((partition_id, label, features), 1)
data_pdf = pd.DataFrame(data_np, columns=["partition_id"] + ["label"] + ["x" + str(x) for x in range(p)])
data_sdf = spark.createDataFrame(data_pdf)

# define a beta schema FIXME: the first two elements of schema are not right. should be
# 'coef', and 'Sig_invMcoef', is "partition_id" and "label"
schema_beta = StructType(
    [StructField('par_id', IntegerType(), True),
     StructField('coef', DoubleType(), True),
     StructField('Sig_invMcoef', DoubleType(), True)]
    + data_sdf.schema.fields[2:])

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

# Repartition
data_sdf = data_sdf.repartition(partition_num, "partition_id")

##----------------------------------------------------------------------------------------
## APPLY USER-DEFINED FUNCTIONS TO PARTITIONED DATA
##----------------------------------------------------------------------------------------

# define the Pandas UDF
@pandas_udf(schema_beta, PandasUDFType.GROUPED_MAP)
def logistic_model(sample_df):
    # run the model on the partitioned data set
    # x_train = sample_df.drop(['label', 'row_id', 'partition_id'], axis=1)
    x_train = sample_df.drop(['partition_id', 'label'], axis=1)
    y_train = sample_df["label"]
    model = LogisticRegression(solver="lbfgs", fit_intercept=False)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_train)[:, 0]
    p = model.coef_.size

    coef = model.coef_.reshape(p, 1) # p-by-1
    Sig_inv = x_train.T.dot(np.multiply((prob*(1-prob))[:,None],x_train)) / prob.size # p-by-p
    Sig_invMcoef = Sig_inv.dot(coef) # p-by-1

    par_id = np.arange(p)

    out_np = np.concatenate((coef, Sig_invMcoef, Sig_inv),1) # p-by-(2+p)
    out_pdf = pd.DataFrame(out_np)
    out = pd.concat([pd.DataFrame(par_id,columns=["par_id"]), out_pdf],1)
    return out

    # return pd.DataFrame(Sig_inv)

# partition the data and run the UDF
mapped_sdf = data_sdf.groupby('partition_id').apply(logistic_model)

# mapped_pdf = mapped_sdf.toPandas()
##----------------------------------------------------------------------------------------
## MERGE
##----------------------------------------------------------------------------------------
groupped_sdf = mapped_sdf.groupby('par_id')
groupped_sdf_sum = groupped_sdf.sum(*mapped_sdf.columns[1:])
groupped_pdf_sum = groupped_sdf_sum.toPandas().sort_values("par_id")

Sig_invMcoef_sum = groupped_pdf_sum.iloc[:,2]
Sig_inv_sum = groupped_pdf_sum.iloc[:,3:]

out_par = np.linalg.solve(Sig_inv_sum,Sig_invMcoef_sum)
out_par_onehot = groupped_pdf_sum['sum(coef)'] / data_sdf.rdd.getNumPartitions()
# out_par_onehot = groupped_pdf_sum['sum(coef)'] / partition_num


##----------------------------------------------------------------------------------------
## MERGE AND DEBIAS with LSA
##----------------------------------------------------------------------------------------
# TOOD: write a native `lars_las()` function
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

def dlsa(Sig_inv, beta, sample_size, intercept=False):

    robjects.r.source("/home/lifeng/code/dlsa/R/dlsa_alasso_func.R")
    lars_lsa=robjects.r['lars.lsa']

    numpy2ri.activate()
    dfitted = lars_lsa(np.array(Sig_inv),np.array(out_par),intercept=False,n=200)
    numpy2ri.deactivate()

    AIC = robjects.FloatVector(dfitted.rx2("AIC"))
    AIC_minIdx = np.argmin(AIC)
    BIC = robjects.FloatVector(dfitted.rx2("BIC"))
    BIC_minIdx = np.argmin(BIC)
    beta = np.array(robjects.FloatVector(dfitted.rx2("beta")))


    if intercept:
        beta0 = np.array(robjects.FloatVector(dfitted.rx2("beta0")) + beta[0])
        beta_AIC = np.concatenate(beta0[AIC_minIdx], beta[AIC_minIdx, :])
        beta_BIC = np.concatenate(beta0[BIC_minIdx], beta[BIC_minIdx, :])
    else:
        beta_AIC = beta[AIC_minIdx, :]
        beta_BIC = beta[BIC_minIdx, :]

    return  pd.DataFrame(beta_AIC, beta_BIC)
