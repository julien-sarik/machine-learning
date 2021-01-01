import pandas as pd
from sklearn.tree import DecisionTreeRegressor

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