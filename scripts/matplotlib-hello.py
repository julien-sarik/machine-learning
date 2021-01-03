import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns


"""
Trend
"""
# 
# line plot
# 
# Read the file into a variable data
data = pd.read_csv("./data/visualization/spotify.csv", index_col="Date", parse_dates=True)
print(data.columns)
plt.figure(figsize=(16,6))
plt.title("Daily Global Streams of Popular Songs in 2017-2018")
plt.xlabel("Date")
sns.lineplot(data=data)
plt.show()

"""
Relationship
"""
#
# scatterplot : to see linear relationship between 2 vars
# 
data = pd.read_csv("./data/visualization/insurance.csv", parse_dates=True)
sns.scatterplot(x=data['bmi'], y=data['charges'], hue=data['smoker'])
plt.show()


"""
distribution
"""
data = pd.read_csv("data/visualization/iris.csv", index_col="Id", parse_dates=True)
print()
print(data.head())
print(data['Species'].unique())
# KDE plot
sns.kdeplot(data=data['Petal Length (cm)'], label="Any", shade=True)
sns.kdeplot(data=data[data.Species == 'Iris-setosa']['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=data[data.Species == 'Iris-versicolor']['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=data[data.Species == 'Iris-virginica']['Petal Length (cm)'], label="Iris-virginica", shade=True)
plt.legend()
plt.show()

# 2D KDE plot
sns.jointplot(x=data['Petal Length (cm)'], y=data['Sepal Width (cm)'], kind="kde")
plt.show()
