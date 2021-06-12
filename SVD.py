import pandas as pd
import numpy as np
import scipy.sparse as sparse
import mysql.connector
import random
from collections import defaultdict
import time

import surprise
from surprise import SVDpp
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split
from surprise import accuracy

def imprimir_data(data):
    f = open("predictions_surprise.txt", "a")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data, file=f)
    f.close()

def get_data():
    try:
        mydb = mysql.connector.connect(host="localhost", database = 'delfos',user="root", passwd="root",use_pure=True)
        query = " SELECT id_cliente uid, cod_interno iid, preferencia rating "
        query+= " FROM cliente_item_preferencia limit 100; "
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close()
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

t0 = time.time()
df = get_data()


# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0.1, 1))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['uid', 'iid', 'rating']], reader)

# test set is made of 25% of the ratings.
trainset, testset = train_test_split(data, test_size=.30)

#algo = SVDpp()
algo = SVD()

# Train the algorithm on the trainset, and predict ratings for the testset
print("Training")
algo.fit(trainset)

print("Testing")
predictions = algo.test(testset)

print("Recommending")
top_n = get_top_n(predictions, n=10)

print("Printing")
# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# Then compute RMSE
accuracy.rmse(predictions, verbose=True)



#cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



'''

TUNING
param_grid = {'n_epochs': [10, 15], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])
# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
print(gs.best_estimator['rmse'])

#RESULTADO: {'n_epochs': 15, 'lr_all': 0.005, 'reg_all': 0.6}
'''




t1 = time.time()
print("Tiempo transcurrido desde el inicio: ",t1-t0)