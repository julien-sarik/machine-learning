import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
Each model is a Classification And Regression Tree (CART). 
While a leaf of a decision tree contains a prediction, the leaf of a CART contains a score. The final score is obtain by summing all scores.
At each iteration the value of a loss function is computed to fit a new model to be added in the ensemble.
"""

# 
# Get and prepare data
# 
# Read the data
data = pd.read_csv('../data/data.csv')
# handle missing values
data = data.dropna(subset=['Price'])
# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
# Select target
y = data.Price
# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

"""
n_estimators specifies how many times to go through the modeling cycle described above. 
It is equal to the number of models that we include in the ensemble.

early_stopping_rounds offers a way to automatically find the ideal value for n_estimators. 
Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators. 
It's smart to set a high value for n_estimators and then use early_stopping_rounds to find the optimal time to stop iterating.
Since random chance sometimes causes a single round where validation scores don't improve, you need to specify a number for how many rounds of straight deterioration to allow before stopping. 
Setting early_stopping_rounds=5 is a reasonable choice. 
In this case, we stop after 5 straight rounds of deteriorating validation scores.

When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.

"""
my_model = XGBRegressor(n_estimators=500)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)],
             verbose=False)
predictions = my_model.predict(X_valid)
print("Mean Absolute Error: {}".format(mean_absolute_error(predictions, y_valid)))