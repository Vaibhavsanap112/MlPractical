#assignment No:05

import pandas as pd 

import numpy as np 


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
df=pd.read_csv('5.csv')
df.head()
df.info()
x=df.drop('Outcome',axis=1)
y=df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
tr=DecisionTreeClassifier(max_depth=3)
tr.fit(x_train,y_train)
y_pred = tr.predict(x_test)
y_pred
metrics.accuracy_score(y_test,y_pred)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

"""# Plot the Decision Tree
plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
plot_tree(tr, feature_names=x.columns, class_names=['Non-Diabetic', 'Diabetic'], filled=True)
plt.show()"""


# Convert DataFrame index to list
feature_names = list(x.columns)

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(tr, feature_names=feature_names, class_names=['Non-Diabetic', 'Diabetic'], filled=True)
plt.show()
