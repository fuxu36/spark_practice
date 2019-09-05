'''
Test the connection of pyspark with s3 bucket locally
'''

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
conf = SparkConf().setAppName("first")
sc = SparkContext(conf=conf)

sc._jsc.hadoopConfiguration().set("fs.s3.awsAccessKeyId", '')
sc._jsc.hadoopConfiguration().set("fs.s3.awsSecretAccessKey", '')
sc._jsc.hadoopConfiguration().set("fs.s3.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")

spark = SparkSession(sc)

a = spark.read.csv('')

