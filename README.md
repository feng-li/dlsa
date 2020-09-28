# `dlsa:` Distributed Least Squares Approximation 
_implemented with Apache Spark_

## Introduction

In this work, we develop a distributed least squares approximation (DLSA) method that is able to solve a large family of regression problems (e.g., linear regression, logistic regression, and Cox's model) on a distributed system. By approximating the local objective function using a local quadratic form, we are able to obtain a combined estimator by taking a weighted average of local estimators. The resulting estimator is proved to be statistically as efficient as the global estimator. Moreover, it requires only one round of communication. We further conduct shrinkage estimation based on the DLSA estimation using an adaptive Lasso approach. The solution can be easily obtained by using the LARS algorithm on the master node. It is theoretically shown that the resulting estimator possesses the oracle property and is selection consistent by using a newly designed distributed Bayesian information criterion (DBIC). The finite sample performance and the computational efficiency are further illustrated by an extensive numerical study and an airline dataset. 

- The entire methodology has been implemented in a Spark system available at https://github.com/feng-li/dlsa. 
- An R package `dlsa` provides the conceptual demo available at https://github.com/feng-li/dlsa_r.

## System Requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `pyarrow >= 0.15.0` Please read this [compatible issue with Spark 2.3.x or 2.4.x](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html#compatibility-setting-for-pyarrow--0150-and-spark-23x-24x)
  - `scikit-learn >= 0.21.2`
  - `rpy2 >= 3.0.4` (optional)

- `R >= 3.5` (optional)
  - `lars`

  See [`setup.py`](setup.py) for detailed requirements.

## Run the [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) code on the Spark platform
```sh
  ./bash/spark_dlsa_run.sh
 ```
 or simply run

 ```py
   ./logistic_dlsa.py
 ```

## References

- [Zhu, X.](https://xueningzhu.github.io/), [Li, F.](http://feng.li/), & [Wang, H.](http://hansheng.gsm.pku.edu.cn/), (2019) Least Squares Approximation for a Distributed System. [_Working Paper_](https://arxiv.org/abs/1908.04904).
