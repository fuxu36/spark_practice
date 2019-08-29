from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
import pandas as pd

'''
Given a target column and a dataset as input, compute the correlation between that
target column and all the features in the dataset.
'''

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('/Users/xue/Desktop/Farrago/Datasets/house price demo/kc_house_data.csv',
                     header=True,
                     inferSchema=True)


def correlation(df, target_col):
    # drop string columns
    columns_to_drop = [item[0] for item in df.dtypes if item[1].startswith('string')]
    df_numeric = df.drop(*columns_to_drop)

    # generate correlation matrix
    features = df_numeric.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(features, method="pearson")
    corr_df = pd.DataFrame(corr_mat)
    corr_df.index, corr_df.columns = df_numeric.columns, df_numeric.columns
    corr_df = corr_df[target_col]

    return corr_df


print(correlation(df, 'price'))



