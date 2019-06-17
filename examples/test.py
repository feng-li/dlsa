#! /usr/bin/env python3

import os

print(os.path.dirname(os.path.abspath(__file__)))


arg1 = "test arg"

def x(arg2):
    if arg2 > 0:
        print(arg1)


x(5)



from pyspark.sql.functions import pandas_udf, PandasUDFType
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)  # doctest: +SKIP
def normalize(pdf):
    v = pdf.v
    return pdf.assign(v=(v - v.mean()) / v.std())
    
df.groupby("id").apply(normalize).show()  # doctest: +SKIP
