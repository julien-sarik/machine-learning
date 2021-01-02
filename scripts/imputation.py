from sklearn.impute import SimpleImputer

"""
Imputation consist in replacing missing values from the training set.
A simple approach is to replace a missing value by the mean.
"""

imputer = SimpleImputer(missing_values=-1, strategy='mean')
print(imputer.fit_transform([[0, -1, 2], [2, 3, 4], [-1, 7, 4]]))