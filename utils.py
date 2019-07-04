#! /usr/bin/env python3

import pandas as pd
import sys

def clean_airlinedata(file_path):
    '''Function to clean airline data from


    http://stat-computing.org/dataexpo/2009/the-data.html
    '''

    pdf0 = pd.read_csv(file_path, usecols = [1,2,3,4,5,6,7,8,9,11,12,14,15,16,17,18,21,23])
    pdf = pdf0.dropna()

    # X_continuous = pdf[['Year', 'DayofMonth', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime',
    #                     'ActualElapsedTime', 'CRSElapsedTime', 'DepDelay', 'Distance']]
    X_with_dummies = pd.get_dummies(data=pdf,
                                    columns=['Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
                                    sparse=True)
    X = X_with_dummies.drop('ArrDelay',axis = 1)
    Y = pdf['ArrDelay']>0 # FIXME: 'Cancelled' 'Diverted' could be used for multilevel logistic

    out_pdf = pd.concat([Y, X], axis=1)

    return out_pdf

def insert_partition_id_pdf(data_pdf, partition_num, partition_method):
    '''Insert arbitrary index to Pandas DataFrame for partition

    '''

    nrow = data_pdf.shape[0]

    if partition_method == "systematic":
        partition_id = pd.DataFrame(pd.RangeIndex(nrow) % partition_num, columns=['partition_id'])
        out = pd.concat([partition_id, data_pdf], axis=1, join_axes=[partition_id.index])

    return out


def insert_partition_id_sdf(data_sdf, partition_num, partition_method):
    ''''Insert arbitrary index to Spark DataFrame for partition

    assign a row ID and a partition ID using Spark SQL
    FIXME: WARN WindowExec: No Partition Defined for Window operation! Moving all data to a
    single partition, this can cause serious performance
    degradation. https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html

    '''
    data_sdf.createOrReplaceTempView("data_sdf")
    data_sdf = spark.sql("""
    select *, row_id%20 as partition_id
    from (
    select *, row_number() over (order by rand()) as row_id
    from data_sdf
    )
    """)


    return data_sdf
