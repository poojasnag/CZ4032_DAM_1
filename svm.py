
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import cross_val_score

names = pd.read_csv('datasets/iris.names', header=None)
header = list(names.iloc[0,])
data = pd.read_csv('datasets/iris.data', names = header)

y = pd.DataFrame(data["class"])
X = pd.DataFrame(data[header[:-1]])

model=SVC(gamma='scale')
model.fit(X, y.values.ravel())

scores = cross_val_score(model, X, y.values.ravel(), cv=10)

print(scores.mean())






