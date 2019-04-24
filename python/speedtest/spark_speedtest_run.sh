#! /usr/bin/sh


# https://help.aliyun.com/document_detail/28124.html

spark-submit \
    --master yarn          \
    --driver-memory 30g    \ # Master's RAM
    --executor-memory 5g   \ # Worker's RAM
    --num-executors 2      \
    spark_speedtest.py


exit 0;
