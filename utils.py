#! /usr/bin/env python3

import pandas as pd
import sys



path = '/home/lifeng/running/data/1987.csv.bz2'


def clean_airlinedata(file_path):
    '''Function to clean airline data from


    http://stat-computing.org/dataexpo/2009/the-data.html
    '''

    pdf0 = pd.read_csv(path, usecols = [0,1,2,3,4,5,6,7,8,9,11,12,14,15,16,17,18,21,23])
    pdf = pdf0.dropna()

    X_continuous = pdf[['Year', 'DayofMonth', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
                        'ActualElapsedTime', 'CRSElapsedTime', 'DepDelay', 'Distance']]
    X_dummies = pd.get_dummies(pdf, columns = ['Month', 'UniqueCarrier', 'Origin', 'Dest'])
    Y = pdf['ArrDelay']>0 # # FIXME: 'Cancelled' 'Diverted' could be used for multilevel logistic

    out_pdf = pd.concat([Y, X_continuous, X_dummies], 1)

    return out_pdf

def add_partition_id(pdf, partition_num, partition_method):
    '''Add arbitrary index for partition
    '''

    nrow = pdf.shape[0]

    if partition_method == "systematic":
        partition_id = pd.RangeIndex(nrow)
        out = pd.concat([pd.DataFrame(partition_id, columns='partition_id'), pdf], 1)


    return out
