import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

names = pd.read_csv('datasets/iris.names', header=None)
header = list(names.iloc[0,])
data = pd.read_csv('datasets/iris.data', names = header)

y = pd.DataFrame(data["class"])
X = pd.DataFrame(data[header[:-1]])

from sklearn.ensemble import RandomForestClassifier

rforest = RandomForestClassifier(random_state=2)  

# Fit Random Forest on data
rforest.fit(X, y.values.ravel())

y_pred = rforest.predict(X)
scores = cross_val_score(rforest, X, y.values.ravel(), cv=10)
print("average accuracy after 10-fold CV: ", scores.mean())
