#! /bin/bash


# https://help.aliyun.com/document_detail/28124.html

for i in {10..100..2}
do
    cat \
        spark_speedtest.py   \
        > speedtest.s$i
done

exit 0;
