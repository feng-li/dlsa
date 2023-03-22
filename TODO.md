
# Known issues

- Problem running with Spark 2.4.1 on some platforms: `Pandas generates null values`. See this example

``` py
import numpy as np
import pandas as pd
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

from pyspark.sql.functions import pandas_udf, PandasUDFType

df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0),(2, 7.0), (2, 10.0)],
    ("id", "v"))

@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())

df.groupby("id").apply(subtract_mean).show()
```

- Problem with Python (< 3.7): `SyntaxError: more than 255 arguments` if p is too large
  with pandas udf. 
  
   - Update Python to 3.7 resolves this problem.
