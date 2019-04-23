#! /usr/bin/sh


# https://help.aliyun.com/document_detail/28124.html

spark-submit \
    --master yarn \
    spark_speedtest.py

exit 0;
