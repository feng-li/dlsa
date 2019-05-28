#! /usr/bin/env python3

import findspark
findspark.init("/usr/lib/spark-current")

import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

from sklearn.linear_model import LogisticRegression

spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
#spark.conf.set("spark.sql.shuffle.partitions", 10)
spark.conf.get("spark.sql.shuffle.partitions")

# load the CSV as a Spark data frame
pandas_df = pd.read_csv("../data/games-expand.csv")
spark_df = spark.createDataFrame(pandas_df)

# assign a user ID and a partition ID using Spark SQL
# FIXME: WARN WindowExec: No Partition Defined for Window operation! Moving all data to a
# single partition, this can cause serious performance
# degradation. https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html
spark_df.createOrReplaceTempView("spark_df")
spark_df = spark.sql("""
select *, user_id%10 as partition_id
from (
  select *, row_number() over (order by rand()) as user_id
  from spark_df
)
""")



# define a schema for the result set, the user ID and model prediction
schema = StructType([StructField('user_id', LongType(), True),
                     StructField('prediction', DoubleType(), True)])

# define the Pandas UDF
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_model(sample_df):

    # run the model on the partitioned data set
    ids = sample_df['user_id']
    x_train = sample_df.drop(['label', 'user_id', 'partition_id'], axis=1)
    y_train = sample_df["label"]
    model = LogisticRegression(solver="lbfgs", fit_intercept=False)
    model.fit(x_train, y_train)
    pred = model.predict_proba(x_train)

    return pd.DataFrame({'user_id': ids, 'prediction': pred[:,1]})

# partition the data and run the UDF
results = spark_df.groupby('partition_id').apply(apply_model)
results.collect()
