# `dlsa`
Distributed Least Squares Approximation implemented with Apache Spark

# System Requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `pyarrow >= 0.15.0` Please read this [compatible issue with Spark 2.3.x or 2.4.x](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
  - `scikit-learn >= 0.21.2`
  - `rpy2 >= 3.0.4` (optional)

- `R >= 3.5` (optional)
  - `lars`

  See [`setup.py`](setup.py) for detailed requirements.

# Run the [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) code on the Spark platform
```sh
  ./bash/spark_dlsa_run.sh
 ```
 or simply run

 ```py
   ./logistic_dlsa.py
 ```

# References

- [Zhu, X.](https://xueningzhu.github.io/), [Li, F.](http://feng.li/), & [Wang, H.](http://hansheng.gsm.pku.edu.cn/), (2019) Least Squares Approximation for a Distributed System. [_Working Paper_](https://arxiv.org/abs/1908.04904).
