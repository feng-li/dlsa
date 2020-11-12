from pyspark.sql import Column
from pyspark.sql.functions import col, count as sparkcount, when, lit
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer, IndexToString
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import json


def withMeta(self, alias, meta):
    sc = spark.sparkContext._active_spark_context
    jmeta = sc._gateway.jvm.org.apache.spark.sql.types.Metadata
    return Column(
        getattr(self._jc, "as")(alias, jmeta.fromJson(json.dumps(meta))))

def get_sdummies(sdf, dummy_columns, keep_top=.01, replace_with='other'):
    """
    Index string columns and group all observations that occur in less then a keep_top% of the rows in sdf per column.
    :param sdf: A pyspark.sql.dataframe.DataFrame
    :param dummy_columns: String columns that need to be indexed
    :param replace_with: String to use as replacement for the observations that need to be grouped.

    Modified based on
    https://stackoverflow.com/questions/48566982/how-to-efficiently-group-levels-with-low-frequency-counts-in-a-high-cardinality
    """
    total = sdf.count()
    for string_col in dummy_columns:
        # Apply string indexer
        pipeline = Pipeline(stages=[
            StringIndexer(inputCol=string_col, outputCol="ix_" + string_col)
        ])
        sdf = pipeline.fit(sdf).transform(sdf)

        # Calculate the number of unique elements to keep
        n_to_keep = sdf.groupby(string_col).agg(
            (sparkcount(string_col) /
             total).alias('perc')).filter(col('perc') > keep_top).count()

        # If elements occur below (keep_top * number of rows), replace them with n_to_keep.
        this_meta = sdf.select('ix_' + string_col).schema.fields[0].metadata
        if n_to_keep != len(this_meta['ml_attr']['vals']):
            this_meta['ml_attr']['vals'] = this_meta['ml_attr']['vals'][0:(
                n_to_keep + 1)]
            this_meta['ml_attr']['vals'][n_to_keep] = replace_with
            sdf = sdf.withColumn(
                'ix_' + string_col,
                when(col('ix_' + string_col) >= n_to_keep,
                     lit(n_to_keep)).otherwise(col('ix_' + string_col)))

        # add the new column with correct metadata, remove original.
        sdf = sdf.withColumn('ix_' + string_col,
                             withMeta(col('ix_' + string_col), "", this_meta))

    return sdf


if __name__ == "__main__":

    # SAMPLE DATA -----------------------------------------------------------------
    import pyspark
    conf = pyspark.SparkConf().setAppName("Spark DLSA App")
    spark = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()

    import pandas as pd
    df = pd.DataFrame({
        'x1': ['a', 'b', 'a', 'b', 'c'],  # a: 0.4, b: 0.4, c: 0.2
        'x2': ['a', 'b', 'a', 'b', 'a'],  # a: 0.6, b: 0.4, c: 0.0
        'x3': ['a', 'a', 'a', 'a', 'a'],  # a: 1.0, b: 0.0, c: 0.0
        'x4': ['a', 'b', 'c', 'd', 'e']
    })  # a: 0.2, b: 0.2, c: 0.2, d: 0.2, e: 0.2
    sdf = spark.createDataFrame(df)

    # TEST THE FUNCTION -----------------------------------------------------------
    sdf = get_sdummies(sdf, sdf.columns, 0.25)

    ix_cols = [x for x in sdf.columns if 'ix_' in x]
    for string_col in ix_cols:
        idx_to_string = IndexToString(inputCol=string_col,
                                      outputCol=string_col[3:] + 'grouped')
        sdf = idx_to_string.transform(sdf)

    sdf.show()
