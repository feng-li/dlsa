#! /usr/bin/env python3

import pandas as pd
import numpy as np
import sys

def clean_airlinedata(file_path, fit_intercept):
    '''Function to clean airline data from


    http://stat-computing.org/dataexpo/2009/the-data.html
    0Year,1Month,2DayofMonth,3DayOfWeek,4DepTime,5CRSDepTime,6ArrTime,7CRSArrTime,8UniqueCarrier,
    9FlightNum, 10TailNum,11ActualElapsedTime,12CRSElapsedTime,13AirTime,14ArrDelay,15DepDelay,
    16Origin,17Dest,18Distance,19TaxiIn, 20TaxiOut,21Cancelled,22CancellationCode,23Diverted,
    24CarrierDelay,25WeatherDelay,26NASDelay,27SecurityDelay,28LateAircraftDela

    Variables `10TailNum，22CancellationCode，24CarrierDelay，25WeatherDelay，26NASDelay，
    27SecurityDelay，28LateAircraftDelay` containing too many NAs. Deleted.

    13AirTime=ActualElapsedTime-TaxiIn-TaxiOut, TaxiIn and TaxiOut only available since 1995.

    '''

    pdf0 = pd.read_csv(file_path, error_bad_lines=False,
                       usecols = [1,2,3,4,5,6,7,8,11,13,14,15,16,17,18],
                       engine='c', # The C engine is faster
                       dtype={'Year': 'Int64',
                              'Month': 'Int64',
                              'DayofMonth': 'Int64',
                              'DayOfWeek': 'Int64',
                              'DepTime': np.float64,
                              'CRSDepTime': np.float64,
                              'ArrTime': np.float64,
                              'CRSArrTime': np.float64,
                              'UniqueCarrier': 'str',
                              'ActualElapsedTime': np.float64,
                              'Origin': 'str',
                              'Dest': 'str',
                              'Distance': np.float64}
    )
    pdf = pdf0.dropna()

    X_with_dummies = pd.get_dummies(data=pdf, drop_first=fit_intercept,
                                    columns=['Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'], # 2, 4, 9, 17, 18
                                    sparse=True)
    X = X_with_dummies.drop('ArrDelay',axis = 1)

    # Obtain labels
    # FIXME: 'Cancelled' 'Diverted' could be used for multilevel logistic
    Y = (pdf['ArrDelay']>0).astype(int)

    out_pdf = pd.concat([Y, X], axis=1).reset_index(drop=True)

    return out_pdf


def get_dummy_keys(dummy_set, fit_intercept):
    ''''Make a `distributed` dummy key list to create a complete data matrix

    dummy_set should be a previous saved pickle file containing all possible dummy set.

    fit_intercept, If True, first level is dropped to produce k-1 dummy_keys

    '''

    with open(os.path.expanduser(dummy_set), "rb") as f:
        column_dumps = pickle.load(f)

    dummy_keys = list(column_dumps.keys())
    dummy_column_names = []

    for i in dummy_keys:
        used_dummy_idx = list(column_dumps[i])[fit_intercept:] # if drop first level
        dummy_column_names.extend([i + '_' + str(x) for x in used_dummy_idx])

    return dummy_column_names



def insert_partition_id_pdf(data_pdf, partition_num, partition_method):
    '''Insert arbitrary index to Pandas DataFrame for partition

    NOTICE: data_pdf should have natural numbers (0,1,2...) as index. Otherwise this will
    mismatch the target and produce NaNs.

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


def convert_schema(schema_fields=None, fields_index=None, out_type=None):
    '''Convert schema type for large data frame

    '''
    if fields_index == None:
        fields_index = range(len(schema_fields))

    for i in fields_index:
        schema_fields[i].dataType = out_type


    return schema_fields
