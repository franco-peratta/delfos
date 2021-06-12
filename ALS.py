import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import scipy.sparse as sparse
import random
import implicit
from datetime import datetime
import time
import mysql.connector
from sklearn import metrics

def imprimir_data(data):
    f = open("data_test.txt", "a")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data, file=f)
    f.close()

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
    test_set = ratings.copy() # Make a copy of the original set to be the test set. 
    #test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
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
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered  

##################
### EVALUACION ### 
##################
def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    parameters:    
        - predictions: your prediction output
        - test: the actual target result you are comparing to    
    returns:   
        - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)

def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered. 
    parameters:    
        - training_set - The training set resulting from make_train, where a certain percentage of the original
        - user/item interactions are reset to zero to hide them from the model 
        - predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
            These should be stored in a list, with user vectors as item zero and item vectors as item one. 
        - altered_users - The indices of the users where at least one user/item pair was altered from make_train function
        - test_set - The test set constucted earlier from make_train function
    
    returns:    
        The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
        there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''   
    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  
   # Return the mean AUC rounded to three decimal places for both test and popularity benchmark

# MAIN
t0 = time.time()
data = get_data()

#imprimir_data(data)

# Creo las matrices (no hacia falta crear las 2)
item_user = sparse.csr_matrix((data['preferencia'].astype('float32'),(data['cod_interno'].astype('int32'),data['id_cliente'].astype('int32'))))
user_item = sparse.csr_matrix((data['preferencia'].astype('float32'),(data['id_cliente'].astype('int32'),data['cod_interno'].astype('int32'))))


product_train, product_test, product_users_altered = make_train(user_item, pct_test = 0.2)

print(product_train.transpose())


'''
# Calculo la "sparsity" de la matriz
matrix_size = item_user.shape[0]*item_user.shape[1] # Number of possible interactions in the matrix
num_purchases = len(item_user.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))
# For collaborative filtering to work, the maximum sparsity you could get away with would probably be about 99.5% or so
print("Sparsity: ",sparsity)
'''

'''
# Building the model
# regularization => lambda
# factors => between 20 and 200
# iterations => the more the better, but it would take longer
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Poner aca que carajo es alpha
alpha_val = 40 # esto es lo recomendado por el paper: In our experiments, α= 40 was found to produce good results.
data_conf = (item_user * alpha_val).astype('double')

print("Training the model")
model.fit(data_conf)

# Get User Recommendations
users = [20074693]
limite_recomendaciones = 5
filtrar_items_comprados = False

for u in users:
    print("Reccomendations for user ", u)
    #recs = model.recommend(u, item_user.transpose(), limite_recomendaciones, filtrar_items_comprados)
    recs = model.recommend(u, user_item, limite_recomendaciones, filtrar_items_comprados)
    print(recs)
'''

'''
# Get similar items
items = [254580,253486] # -> cerveza y azucar ledesma
limite_similares = 5

for i in items:
    print("Similar items to item ", i, ": ")
    similar = model.similar_items(i, limite_similares)
    print(similar)
'''

t1 = time.time()
print("Tiempo transcurrido desde el inicio: ",t1-t0)