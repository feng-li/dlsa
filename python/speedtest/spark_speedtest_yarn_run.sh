#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html
# Fat executors: one executor per node
# EC=2
# EM=32g

# Tiny executors: one executor per core
EC=1
EM=2g

for i in 1 {2..64..4}
# for i in 1 2
do
    spark-submit \
        --master yarn  \
        --driver-memory 30g    \
        --executor-memory $EM   \
        --num-executors $i      \
        --executor-cores $EC    \
        --conf spark.rpc.message.maxSize=1024 \
        spark_speedtest.py   \
        > speedtest.s$i.ec$EC 2> speedtest.s$i.ec$EC.log
done

exit 0;
