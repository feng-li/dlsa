# `dlsa`: Distributed Least Squares Approximation implemented with Apache Spark

# Requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `scikit-learn >= 0.21.2`
  - `rpy2 >= 3.0.4`

- `R >= 3.5`
  - `lars`

  See [`setup.py`](setup.py) for detailed requirements.

# Run the code with cluster
```sh
  ./bash/spark_dlsa_run.sh
 ```
 or simply run

 ```py
   ./logistic_dlsa.py
 ```

# References

- Zhu, X., Wang, H., & Li, F., (2019) Least Squares Approximation for a Distributed System. _Working Paper_.

# Known issues

- Problem running with Spark 2.4.1: Pandas generates null values. See this example

``` py
import numpy as np
import pandas as pd
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

from pyspark.sql.functions import pandas_udf, PandasUDFType

df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
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
