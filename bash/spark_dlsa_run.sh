#! /bin/bash

# https://help.aliyun.com/document_detail/28124.html

# Fat executors: one executor per node
# EC=16
# EM=30g

# Make a zip package
# zip -r ../../dlsa.zip ../dlsa.py ../R/dlsa_alasso_func.R

MODEL_DESCRIPTION=$1

# Tiny executors: one executor per core
EC=1
EM=10g

# MODEL_FILE=logistic_spark
MODEL_FILE=logistic_dlsa
OUTPATH=~/running/
# MODEL_FILE=logistic_SGD

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# for i in 1 {4..100..4} # 1, 5, 10, 15, ... , 100
# for i in {256..4..-4}
for executors in 6
do
    tic=`date +%s`
    PYSPARK_PYTHON=python3 spark-submit \
                  --master yarn  \
                  --driver-memory 50g    \
                  --executor-memory ${EM}   \
                  --num-executors ${executors}      \
                  --executor-cores ${EC}    \
                  --conf spark.rpc.message.maxSize=1024 \
                  $DIR/../${MODEL_FILE}.py  \
                  > ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${executors}.EC${EC}.out 2> ${OUTPATH}${MODEL_DESCRIPTION}_${MODEL_FILE}.NE${executors}.EC${EC}.log
    toc=`date +%s`
    runtime=$((toc-tic))
    echo ${MODEL_FILE}.NE${executors}.EC${EC} done, "Time used (s):" $runtime

done

exit 0;
