# https://towardsdatascience.com/alternating-least-square-for-implicit-dataset-with-code-8e7999277f4b
# https://gist.github.com/himanshk96/21594b9f49a8b3060ff1f00d0a0d8ec5

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import scipy.sparse as sparse
import random
import implicit
from datetime import datetime
import time

def imprimir_data(data):
    f = open("data_test.txt", "a")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data, file=f)
    f.close()

def create_data(datapath,start_date,end_date):
    df=pd.read_csv(datapath)
    df=df.assign(date=pd.Series(datetime.fromtimestamp(a/1000).date() for a in df.timestamp))
    df=df.sort_values(by='date').reset_index(drop=True) # for some reasons RetailRocket did NOT sort data by date
    df=df[(df.date>=datetime.strptime(start_date,'%Y-%m-%d').date())&(df.date<=datetime.strptime(end_date,'%Y-%m-%d').date())]
    df=df[['visitorid','itemid','event']]
    return df

t0 = time.time()

datapath = './events.csv'
data = create_data(datapath,'2015-5-3','2015-5-18')


# 1 -> view
# 2 -> addtocart
# 3 -> transaction
data.loc[data['event'] == "view", 'event'] = 1
data.loc[data['event'] == "addtocart", 'event'] = 2
data.loc[data['event'] == "transaction", 'event'] = 3

data['visitorid'] = data['visitorid'].astype("category")
data['itemid'] = data['itemid'].astype("category")
data['visitor_id'] = data['visitorid'].cat.codes
data['item_id'] = data['itemid'].cat.codes

#data['event']=data['event'].astype('category')
#data['event']=data['event'].cat.codes

data.info()
exit()


# Create the sparse matrix
sparse_item_user = sparse.csr_matrix((data['event'].astype(float), (data['item_id'], data['visitor_id'])))
sparse_user_item = sparse.csr_matrix((data['event'].astype(float), (data['visitor_id'], data['item_id'])))
#print(sparse_item_user)
#print("----------------------------------------")
#print(sparse_user_item)



#Building the model
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

alpha_val = 40
data_conf = (sparse_item_user * alpha_val).astype('double')

model.fit(data_conf)

# Get Recommendations
user_id = 14
recs = model.recommend(user_id, sparse_user_item, 10, True)
print(recs)

#Get similar items
#item_id = 7
#n_similar = 3
#similar = model.similar_items(item_id, n_similar)
#print(similar)

t1 = time.time()
print("Tiempo transcurrido desde el inicio: ",t1-t0)