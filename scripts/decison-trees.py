import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# 
# Read and explore data
# 
# this file contains prices for houses
melbourne_file_path = '../data/data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the DataFrame
print(melbourne_data.describe())
print(melbourne_data.shape)
# print DataFrame columns
print(melbourne_data.columns)
# print top few rows
print(melbourne_data.head())


# 
# Handle missing values
# 
# Number of missing values in each column
missing_val_count_by_column = (melbourne_data.isnull().sum())
print('columns with missing values:')
print(missing_val_count_by_column[missing_val_count_by_column > 0])
# dropna() drops rows with missing values
melbourne_data = melbourne_data.dropna()


# by convention y is the variable name for the prediction target
y = melbourne_data.Price
# features (columns) are the model input
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# by convention X is the variable name for the features
X = melbourne_data[melbourne_features]


# 
# Decision tree
# 
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)
# compute MAE (Mean Absolute Error)
predicted_home_prices = melbourne_model.predict(X)
print('MAE when validating on training data: %f' %mean_absolute_error(y, predicted_home_prices))
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) # note the multiple assignment
# Fit model
melbourne_model.fit(train_X, train_y)
print('with default parameters the decision tree has %d max depth and %d leaves' %(melbourne_model.get_depth(), melbourne_model.get_n_leaves()))
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
# compute MAE (Mean Absolute Error)
print('MAE when validating on validation data: %f' %mean_absolute_error(val_y, val_predictions))

# 
# Customization of the model to avoid underfitting and overfitting
#
# find the best max_leaf_nodes parameter 
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))


"""

Random forest

The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
It generally has much better predictive accuracy than a single decision tree and it works well with default parameters (i.e. without customization of the model).
RF trees are independent from each other therefore learning can leverage parallelization.
"""
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print('MAE with random forest %d' %mean_absolute_error(val_y, melb_preds))