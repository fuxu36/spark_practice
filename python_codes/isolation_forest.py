from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark_iforest.ml.iforest import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.functions import array
from pyspark.sql import SQLContext as sqlContext
from pyspark.ml.feature import VectorAssembler
import tempfile
import time


start = time.time()
#file = '/Users/xue/Desktop/Farrago/Datasets/Netlogx Training Data/part-00190-147ed921-7fa3-436f-928b-51de852735be-c000.csv'
#file = '/Users/xue/Desktop/Farrago/Datasets/Netlogx Training Data/combined_csv.csv'
file = '/Users/xue/Desktop/Farrago/Datasets/house price demo/kc_house_data.csv'

spark = SparkSession \
        .builder.master("local[*]") \
        .appName("IForestExample") \
        .getOrCreate()

data = spark.read.csv(file
                      , header=True
                      , inferSchema=True)
print('Number of Rows: {}'.format(data.count()))

columns_not_str = [item[0] for item in data.dtypes if not item[1].startswith('string')]
print(columns_not_str)

# convert value in columns to an array
# then convert array to vector
df = data.select(array(columns_not_str).alias('features'))
to_vector = udf(lambda a: Vectors.dense(a), VectorUDT())
df = df.select(to_vector("features").alias("features"))
df.show()


# convert value in columns to vector directly
# vecAssembler = VectorAssembler(inputCols=columns_not_str, outputCol="features")
# df = vecAssembler.transform(data)
# df.show()

# Init an IForest Object and Fit on a given data frame
iforest = IForest(maxDepth=10
                  , numTrees=10
                  , bootstrap=False
                  #, approxQuantileRelativeError=0.05
                  # , contamination=0.05
                  , maxSamples=256
                  , maxFeatures = 20)

model = iforest.fit(df)

# Check if the model has summary or not, the newly trained model has the summary info
model.hasSummary
# Show model summary
summary = model.summary
# Show the number of anomalies
print(summary.numAnomalies)
# Predict for a new data frame based on the fitted model
transformed = model.transform(df)

# Collect spark data frame into local df
rows = transformed.select(transformed.columns)
temp_path = tempfile.mkdtemp()
iforest_path = temp_path + "/iforest"

# Save the iforest estimator into the path
iforest.save(iforest_path)

# Load iforest estimator from a path
loaded_iforest = IForest.load(iforest_path)
model_path = temp_path + "/iforest_model"

# Save the fitted model into the model path
model.save(model_path)

# Load a fitted model from a model path and  predict a new data frame
loaded_model = IForestModel.load(model_path)
loaded_model.hasSummary
loaded_model.transform(df).show()

end = time.time()
print(end-start)