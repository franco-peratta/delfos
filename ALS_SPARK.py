import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import mysql.connector
import time

from pyspark.mllib.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as sql_func
from pyspark.sql.types import *
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, explode


def get_data_coope():
    try:
        mydb = mysql.connector.connect(host="localhost", database = 'delfos',user="root", passwd="root",use_pure=True)
        #query = "SELECT * FROM cliente_preferencia limit 1020000;"
        query = "SELECT * FROM cliente_item_preferencia ;"
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close()
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))

def get_data_UK():
    try:
        mydb = mysql.connector.connect(host="localhost", database = 'delfos',user="root", passwd="root",use_pure=True)
        query = "SELECT * FROM preferencias_UK ;"
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close()
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))


# MAIN
t0 = time.time()

#pandas_df = get_data_coope()
pandas_df = get_data_UK()

#https://stackoverflow.com/questions/30763951/spark-context-sc-not-defined
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


data = spark.createDataFrame(pandas_df)

#data.describe().show()

(training, test) = data.randomSplit([0.8, 0.2])
#test = data

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(
    maxIter=5,
    rank = 150,
    regParam=0.1,
    alpha=1.0,
    userCol="id_cliente",
    itemCol="cod_interno",
    ratingCol="preferencia",
    coldStartStrategy="drop",
    implicitPrefs=True
)
'''
EVALUATION
# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="preferencia", 
           predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))

# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

#Fit cross validator to the 'train' dataset
#Extract best model from the cv model above
model_cv = cv.fit(training)
#View the predictions
best_model = model_cv.bestModel 
test_predictions = best_model.transform(test)
RMSE = evaluator.evaluate(test_predictions)

**Best Model**
  Rank: 150
  MaxIter: 5
  RegParam: 0.1
  RMSE: 0.34518970808743255 (decente)

print("**Best Model**")# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())
'''

# Training
model = als.fit(training)


# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each item
itemRecs = model.recommendForAllItems(10)


# Print recommendations
'''
userRecs = userRecs\
    .withColumn("rec_exp", explode("recommendations"))\
    .select('id_cliente', col("rec_exp.cod_interno"), col("rec_exp.preferencia"))
'''
userRecs.limit(100).show()


# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="preferencia",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))



t1 = time.time()
print("Tiempo transcurrido desde el inicio: ",t1-t0)
