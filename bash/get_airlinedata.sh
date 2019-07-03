#! /bin/bash

# Download data from
# http://stat-computing.org/dataexpo/2009/

TARGET_DIR=~/running/data

cd $TARGET_DIR
for year in 1 {1987..2008..1}
do

    wget http://stat-computing.org/dataexpo/2009/${year}.csv.bz2
    echo ${year}.csv.bz2 was downloaded to $TARGET_DIR .

done
