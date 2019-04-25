#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html

# Fat executors: one executor per node
EC=16
EM=80g

# Tiny executors: one executor per core
# EC=1
# EM=5g

for i in 1 {5..100..5} # 1, 5, 10, 15, ... , 100
do
    spark-submit \
        --master spark://${SPARK_MASTER}:7077  \
        --driver-memory 30g    \
        --executor-memory $EM   \
        --num-executors $i      \
        --executor-cores $EC
        --conf spark.rpc.message.maxSize=1024 \
        spark_speedtest.py   \
        > speedtest.s$i.ec$EC 2> speedtest.s$i.ec$EC.log
done

exit 0;
