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

    awk -F, 'NR >= 2' ${SOURCE}${i}.csv >> ${SOURCE}allfile.csv
    echo $i is processed
done


# Insert a header
# sed -i '1 i\Year,Month,DayofMonth,DayOfWeek,DepTime,CRSDepTime,ArrTime,CRSArrTime,UniqueCarrier,FlightNum,TailNum,ActualElapsedTime,CRSElapsedTime,AirTime,ArrDelay,DepDelay,Origin,Dest,Distance,TaxiIn,TaxiOut,Cancelled,CancellationCode,Diverted,CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay' allfile.csv
