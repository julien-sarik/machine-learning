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
# Set the width and height of the figure
# plt.figure(figsize=(16,6))
# plt.title("Daily Global Streams of Popular Songs in 2017-2018")
# Add label for horizontal axis
# plt.xlabel("Date")
# sns.lineplot(data=data)

"""
Relationship
"""
#
# regplot : to see linear relationship between 2 vars
# 


"""
distribution
"""
data = pd.read_csv("data/visualization/iris.csv", index_col="Id", parse_dates=True)
# KDE plot
sns.kdeplot(data=data['Petal Length (cm)'], shade=True)
# 2D KDE plot
sns.jointplot(x=data['Petal Length (cm)'], y=data['Sepal Width (cm)'], kind="kde")
plt.show()
