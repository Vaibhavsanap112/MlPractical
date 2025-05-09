#assignment No:03

#import pandas
import pandas as pd
# load dataset
pima = pd.read_csv("3.csv")
pima.head()
pima.shape
#split dataset in features and target variable
feature_cols = ['age' ,'cigsPerDay', 'prevalentHyp', 'heartRate','totChol','sysBP','diaBP']
X = pima[feature_cols] # Features
y = pima.TenYearCHD # Target variable
# split X and y into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  

class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
# Assuming cnf_matrix is defined elsewhere in your code
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)  # added ax=ax here
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import classification_report
target_names = ['without Heart Disease', 'with Heart Disease']
print(classification_report(y_test, y_pred, target_names=target_names))
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  

# Plot count of 'TenYearCHD' column
sns.countplot(x='TenYearCHD', data=pima, palette='hls')
plt.title('Count of Heart Disease')
plt.xlabel('Heart Disease (0: No, 1: Yes)')
plt.ylabel('Count')

# Save the plot
plt.savefig('count_plot.png')

# Show plot
plt.show()
