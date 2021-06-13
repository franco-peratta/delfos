<<<<<<< HEAD
# Links
# https://jessesw.com/Rec-System/
# https://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/RecEngine_NB.ipynb

import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

# The first step is to load the data in. Since the data is saved in an Excel file, we can use Pandas to load it.
# Pass website_url as a parameter to the following function to download the file
# If already downloaded, pass the route
# website_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
retail_data = pd.read_excel(r'F:\Programas\ProyectoFinalImplicit\ecommerce\Online Retail modified.xlsx') # This may take a couple minutes
#retail_data = pd.read_excel(website_url) # This may take a couple minutes

# Muestro la info de la tabla
# retail_data.info()
# Most columns have no missing values, but Customer ID is missing in several rows. 
# If the customer ID is missing, we don’t know who bought the item. We should drop these rows from our data first. 
# We only keep the rows that have a customer ID.
cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]
# cleaned_retail.info()

# it would be nice to have a lookup table that keeps track of each item ID along with a description of that item
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease

# Now we need to:
#   - Group purchase quantities together by stock code and item ID
#   - Change any sums that equal zero to one (this can happen if items were returned, but we want to indicate that the user actually purchased the item instead of assuming no interaction between the user and the item ever took place)
#   - Only include customers with a positive purchase total to eliminate possible errors
#   - Set up our sparse ratings matrix
# This last step is especially important if you don’t want to have unnecessary memory issues! 
# If you think about it, our matrix is going to contain thousands of items and thousands of users with a user/item value required for every possible combination. 
# That is a LARGE matrix, so we can save a lot of memory by keeping the matrix sparse and only saving the locations and values of items that are not zero.
cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive
print(grouped_purchased)

# Our last step is to create the sparse ratings matrix of users and items utilizing the code below:
customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get our unique customers
products = list(grouped_purchased.StockCode.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.Quantity) # All of our purchases

#
# Esto de abajo no anda todavia VER VER VER, para mi se genera mal purchases_sparses
#

# TypeError: Categorical is not ordered for operation max
# you can use .as_ordered() to change the Categorical to an ordered one 
#rows = grouped_purchased.CustomerID.astype('category', categories = customers).cat.codes
#rows = grouped_purchased.CustomerID.astype(CategoricalDtype(categories=customers, ordered=True))
cols = grouped_purchased.CustomerID.astype(int)
# Get the associated row indices
#cols = grouped_purchased.StockCode.astype('category', categories = products).cat.codes
#cols = grouped_purchased.StockCode.astype(CategoricalDtype(categories=products, ordered=True))
rows = grouped_purchased.StockCode.astype(int)

# Get the associated column indices
#purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
purchases_sparse = sparse.csr_matrix( (quantity, (rows, cols)) )

print(purchases_sparse)



# Check the sparsity of the new matrix
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))
# For collaborative filtering to work, the maximum sparsity you could get away with would probably be about 99.5% or so
print("Sparsity: ",sparsity)


t1 = time.time()
=======
# Links
# https://jessesw.com/Rec-System/
# https://nbviewer.jupyter.org/github/jmsteinw/Notebooks/blob/master/RecEngine_NB.ipynb

import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
import time
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()

# The first step is to load the data in. Since the data is saved in an Excel file, we can use Pandas to load it.
# Pass website_url as a parameter to the following function to download the file
# If already downloaded, pass the route
# website_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
retail_data = pd.read_excel(r'F:\Programas\ProyectoFinalImplicit\ecommerce\Online Retail modified.xlsx') # This may take a couple minutes
#retail_data = pd.read_excel(website_url) # This may take a couple minutes

# Muestro la info de la tabla
# retail_data.info()
# Most columns have no missing values, but Customer ID is missing in several rows. 
# If the customer ID is missing, we don’t know who bought the item. We should drop these rows from our data first. 
# We only keep the rows that have a customer ID.
cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]
# cleaned_retail.info()

# it would be nice to have a lookup table that keeps track of each item ID along with a description of that item
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease

# Now we need to:
#   - Group purchase quantities together by stock code and item ID
#   - Change any sums that equal zero to one (this can happen if items were returned, but we want to indicate that the user actually purchased the item instead of assuming no interaction between the user and the item ever took place)
#   - Only include customers with a positive purchase total to eliminate possible errors
#   - Set up our sparse ratings matrix
# This last step is especially important if you don’t want to have unnecessary memory issues! 
# If you think about it, our matrix is going to contain thousands of items and thousands of users with a user/item value required for every possible combination. 
# That is a LARGE matrix, so we can save a lot of memory by keeping the matrix sparse and only saving the locations and values of items that are not zero.
cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive
print(grouped_purchased)

# Our last step is to create the sparse ratings matrix of users and items utilizing the code below:
customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get our unique customers
products = list(grouped_purchased.StockCode.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.Quantity) # All of our purchases

#
# Esto de abajo no anda todavia VER VER VER, para mi se genera mal purchases_sparses
#

# TypeError: Categorical is not ordered for operation max
# you can use .as_ordered() to change the Categorical to an ordered one 
#rows = grouped_purchased.CustomerID.astype('category', categories = customers).cat.codes
#rows = grouped_purchased.CustomerID.astype(CategoricalDtype(categories=customers, ordered=True))
cols = grouped_purchased.CustomerID.astype(int)
# Get the associated row indices
#cols = grouped_purchased.StockCode.astype('category', categories = products).cat.codes
#cols = grouped_purchased.StockCode.astype(CategoricalDtype(categories=products, ordered=True))
rows = grouped_purchased.StockCode.astype(int)

# Get the associated column indices
#purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(customers), len(products)))
purchases_sparse = sparse.csr_matrix( (quantity, (rows, cols)) )

print(purchases_sparse)



# Check the sparsity of the new matrix
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] # Number of possible interactions in the matrix
num_purchases = len(purchases_sparse.nonzero()[0]) # Number of items interacted with
sparsity = 100*(1 - (num_purchases/matrix_size))
# For collaborative filtering to work, the maximum sparsity you could get away with would probably be about 99.5% or so
print("Sparsity: ",sparsity)


t1 = time.time()
>>>>>>> 0dfa0d4ca3694423052fecb78447c0762361facf
print("Tiempo transcurrido desde el inicio: ",t1-t0)