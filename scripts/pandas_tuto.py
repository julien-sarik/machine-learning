import pandas as pd

melbourne_file_path = '../data/data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
print(melbourne_data.describe())
# print columns
print(melbourne_data.columns)
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# y is the predictio target
# y = melbourne_data.Price

# features (columns) are the model input
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']