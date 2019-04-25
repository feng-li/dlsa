#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html

for i in 1 {5..100..5} # 1, 5, 10, 15, ... , 100
do
    spark-submit \
        --master spark://${SPARK_MASTER}:7077  \
        --driver-memory 30g    \
        --executor-memory 30g   \
        --num-executors $i      \
        --conf spark.rpc.message.maxSize=1024 \
        spark_speedtest.py   \
        > speedtest.s$i 2> speedtest.s$i.log
done

exit 0;
