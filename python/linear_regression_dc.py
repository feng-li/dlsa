#! /usr/bin/python3

import findspark
findspark.init("/usr/lib/spark-current")
import pyspark

from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, lit
import statsmodels.api as sm
import pandas as pd

# df has four columns: id, y, x1, x2
# https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html
spark = pyspark.sql.SparkSession.builder.appName("Spark Machine Learning App").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")


df = spark.range(0, 10 * 1000 * 1000).withColumn('id', (col('id') / 10000).cast('integer')).withColumn('v', rand())
df2 = df.withColumn('y', rand()).withColumn('x1', rand()).withColumn('x2', rand()).select('id', 'y', 'x1', 'x2')

group_column = 'id'
y_column = 'y'
x_columns = ['x1', 'x2']
schema = df2.select(group_column, *x_columns).schema

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
# Input/output are both a pandas.DataFrame
def ols(pdf):
    group_key = pdf[group_column].iloc[0]
    y = pdf[y_column]
    X = pdf[x_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    return pd.DataFrame([[group_key] + [model.params[i] for i in   x_columns]],
                        columns=[group_column] + x_columns)

beta = df2.groupby(group_column).apply(ols)
beta.collect()
