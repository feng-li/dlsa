#! /usr/bin/sh


# https://help.aliyun.com/document_detail/28124.html

spark-submit \
    --master yarn \
    --driver-memory 30g \
    --executor-memory 5g \
    spark_speedtest.py


exit 0;
