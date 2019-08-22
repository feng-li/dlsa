# `dlsa`: Distributed Least Squares Approximation implemented with Apache Spark

# System Requirements

- `Spark >= 2.3.1`
- `Python >= 3.7.0`
  - `scikit-learn >= 0.21.2`
  - `rpy2 >= 3.0.4`

- `R >= 3.5`
  - `lars`

  See [`setup.py`](setup.py) for detailed requirements.

# Run the code on the Spark platform
```sh
  ./bash/spark_dlsa_run.sh
 ```
 or simply run

 ```py
   ./logistic_dlsa.py
 ```

# References

- [Zhu, X.](https://xueningzhu.github.io/), [Li, F.](http://feng.li/), & [Wang, H.](http://hansheng.gsm.pku.edu.cn/), (2019) Least Squares Approximation for a Distributed System. [_Working Paper_](https://arxiv.org/abs/1908.04904).
