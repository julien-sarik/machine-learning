import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# this file contains prices for houses
melbourne_file_path = '../data/data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the DataFrame
print(melbourne_data.describe())
# print DataFrame columns
print(melbourne_data.columns)
# print top few rows
print(melbourne_data.head())
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# by convention y is the variable name for the prediction target
y = melbourne_data.Price

# features (columns) are the model input
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
# by convention X is the variable name for the features
X = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))

# compute MAE
predicted_home_prices = melbourne_model.predict(X)
mae = mean_absolute_error(y, predicted_home_prices)
print('MAE when validating on training data: ' + str(mae))


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor(random_state=1)
# Fit model
melbourne_model.fit(train_X, train_y)
print('with default parameters the decision tree has %d max depth and %d leaves' %(melbourne_model.get_depth(), melbourne_model.get_n_leaves()))
# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print('MAE when validating on validation data: %f' %mean_absolute_error(val_y, val_predictions))

# 
# Customization of the model to avoid underfitting and overfitting
# 
for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, mae))

# 
# Random forest
#
# The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
# It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print('MAE with random forest %d' %mean_absolute_error(val_y, melb_preds))