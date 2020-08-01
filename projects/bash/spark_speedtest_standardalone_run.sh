#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html

# Fat executors: one executor per node
EC=16
EM=80g

# Tiny executors: one executor per core
# EC=1
# EM=5g

MODEL_FILE=logistic_LBFGS.p

# for i in 1 {4..100..4} # 1, 5, 10, 15, ... , 100
for i in 1 2
do
    spark-submit \
        --master spark://${SPARK_MASTER}:7077  \
        --driver-memory 30g    \
        --executor-memory $EM   \
        --num-executors $i      \
        --executor-cores $EC    \
        --conf spark.rpc.message.maxSize=1024 \
        $MODEL_FILE  \
        > $MODEL_FILE.NE$i.EC$EC.out 2> $MODEL_FILE.NE$i.EC$EC.log
done

exit 0;
