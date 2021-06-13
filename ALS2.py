import implicit
from implicit.evaluation import precision_at_k
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import random
import time
import mysql.connector
from sklearn import metrics

def get_data():
    try:
        mydb = mysql.connector.connect(host="localhost", database = 'delfos',user="root", passwd="root",use_pure=True)
        #query = "SELECT * FROM cliente_preferencia limit 1020000;"
        query = "SELECT * FROM cliente_preferencia limit 500000;"
        result_dataFrame = pd.read_sql(query,mydb)
        mydb.close()
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))

####################
### TRAINING SET ### 
####################
def make_train(ratings, pct_test = 0.2):
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set. 
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, list(set(user_inds)) # Output the unique list of user rows that were altered  


# MAIN
data = get_data()

# Creo la matriz user-item
testing_set = sparse.csr_matrix((data['preferencia'].astype('float32'),(data['id_cliente'].astype('int32'),data['cod_interno'].astype('int32'))))
training_set, product_users_altered = make_train(testing_set, pct_test = 0.2)


# Building the model
# regularization => lambda
# factors => between 20 and 200
# iterations => the more the better, but it would take longer
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=1, iterations=10, calculate_training_loss=False)

# Poner aca que carajo es alpha
alpha_val = 40.0 # esto es lo recomendado por el paper: In our experiments, α= 40 was found to produce good results.
data_conf = (training_set.transpose() * alpha_val).astype('float32')

print("Training the model")
model.fit(data_conf)

print("Precision@k")
p = precision_at_k(model, training_set, testing_set, K=5, num_threads=4)

print(p)
'''
# Get User Recommendations
users = [20074693]
limite_recomendaciones = 5
filtrar_items_con_interaccion = True

for u in users:
    print("Reccomendations for user ", u)    
    recs = model.recommend(u, testing_set, limite_recomendaciones, filtrar_items_con_interaccion)
    print(recs)

# tarda mucho
# model.recommend_all(testing_set, limite_recomendaciones, filtrar_items_con_interaccion, show_progress=True)

'''

# 0.03858843781380003