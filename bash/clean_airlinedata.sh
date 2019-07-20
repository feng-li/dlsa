#! /usr/bin/env bash

SOURCE=~/running/data_raw/

# Get current dir path for this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


for i in {1987..2007..1}
do

    # if [ $i = '1987' ]
    # then
    #     lines=1
    # else
    #     lines=2
    # fi

    awk -F, 'NR >= 2' ${SOURCE}${i}.csv >> ${SOURCE}allfile_sorted_no_head.csv
    echo $i is processed
done

# Extract dummy columns
awk -F ','  '{printf ("%s,%s,%s,%s,%s,%s\n", $1,$2,$4,$9,$17,$18)}' allfile_ordered_no_head.csv | xz > dummies.xz

# Shuffle lines
shuf allfile_ordered_no_head.csv > allfile_shuffle_no_head.csv

# Split the big file into small files (l will not break lines)
split -n l/21 --additional-suffix=.csv allfile_shuffle_no_head.csv

# Insert a header and compress to bz2 format
for file in xa*.csv
do

    sed -i '1 i\Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay' $file

    bzip2 $file

done
