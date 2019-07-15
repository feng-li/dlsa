#! /usr/bin/env bash

SOURCE=~/running/data_raw/

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


for i in {1987..2007..1}
do

    if [i -eq 1987]
    then
        lines='NR != 0'
    else
        lines='NR != 1'

    awk -F, $lines ${SOURCE}${i}.csv >> ${SOURCE}allfile.csv
done
