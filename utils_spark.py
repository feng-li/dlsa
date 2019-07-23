#! /usr/bin/env python3

from pyspark.sql.types import *


def convert_schema(usecols_x, dummy_info, fit_intercept):
    '''Convert schema type for large data frame

    '''

    schema_fields = []
    if len(dummy_info) == 0: # No dummy is used
        for j in usecols_x:
            schema_fields.append(StructField(j, DoubleType(), True))

    else:
        # Use dummy
        convert_dummies = list(dummy_info['factor_selected'].keys())

        for x in list(set(usecols_x) - set(convert_dummies)):
            schema_fields.append(StructField(x, DoubleType(), True))

        for i in convert_dummies:
            for j in dummy_info["factor_selected_names"][i][fit_intercept:]:
                schema_fields.append(StructField(j, DoubleType(), True))

    return schema_fields


def clean_airlinedata_sdf():

    usecols = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
               'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'ActualElapsedTime', 'AirTime',
               'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance']

    sdfraw0=spark.read.csv(file_path_hdfs[file_no_i],header=True)
    sdf0 = sdfraw0.select(usecols)
    sdf0.dropna()


    data_sdf.createOrReplaceTempView("data_sdf")
    data_sdf = spark.sql(
        """
        select *, row_id%20 as partition_id
        from (
        select *, row_number() over (order by rand()) as row_id
        from data_sdf
        )
        """
    )



    from pyspark.sql import functions



    return False


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
