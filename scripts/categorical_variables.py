import pandas as pd
from sklearn.preprocessing import OneHotEncoder

X = [['Male', 1], ['Female', 3], ['Female', 2]]
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X)
print(encoder.categories_)
# categories are Female, Male, 1, 2 ,3

encoded_matrix = encoder.transform([['Female', 1], ['Male', 4]]).toarray()
# respective columns of the resulting matrix means: is a Female ? is a Male ? is a 1 ? is a 2 ? is a 3 ?
print(encoded_matrix)