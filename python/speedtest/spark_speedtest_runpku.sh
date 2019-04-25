#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html

for i in {1..100..2}
do
    spark-submit \
        --master spark://${SPARK_MASTER}:7077  \
        --driver-memory 120g    \
        --executor-memory 120g   \
        --num-executors $i      \
        spark_speedtest.py   \
        > speedtest.s$i
done

exit 0;
