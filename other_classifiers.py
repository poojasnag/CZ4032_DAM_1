import argparse
import numpy as np
from numpy.core.numeric import cross
import pandas as pd

from sklearn.metrics import make_scorer, accuracy_score, log_loss, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
import warnings

# from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--filename', "-f", default="iris", help="Dataset name")

class proba_logreg(LogisticRegression):
    def predict(self, X):
        return LogisticRegression.predict_proba(self, X)

def preprocess_data(dataset):
    # le = preprocessing.LabelEncoder()
    names = pd.read_csv(f'datasets/{dataset}.names', header=None)
    header = list(names.iloc[0,])
    data = pd.read_csv(f'datasets/{dataset}.data', names = header)

    y = pd.DataFrame(data.iloc[:,-1:])
    X = pd.DataFrame(data[header[:-1]])
    return X, y

def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":

    args = parser.parse_args()
    print('Dataset: ', args.filename)
    X, y = preprocess_data(args.filename)

    print(f"========================== Decision Tree ==========================")
    originalclass = []
    predictedclass = []
    dectree = DecisionTreeClassifier(random_state=2)
    dectree.fit(X, y.values.ravel())
    dectree_results = cross_validate(dectree, X, y.values.ravel(), cv=10, scoring=make_scorer(classification_report_with_accuracy_score))

    print("Average Accuracy: ", dectree_results['test_score'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))



    print(f"========================== Random Forest ==========================")
    originalclass = []
    predictedclass = []
    rforest = RandomForestClassifier(random_state=2)
    rforest.fit(X, y.values.ravel())
    rf_results = cross_validate(rforest, X, y.values.ravel(), cv=10, scoring=make_scorer(classification_report_with_accuracy_score))
    print("Average Accuracy: ", rf_results['test_score'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))
    # print("Average Micro F1 Score: ",f1_score(originalclass, predictedclass, average='micro'))
    # print("Average Macro F1 Score: ",f1_score(originalclass, predictedclass, average='macro'))
    # print("Average Weighted F1 Score: ",f1_score(originalclass, predictedclass, average='weighted'))

    print()
    print(f"========================== SVM ==========================")

    originalclass = []
    predictedclass = []
    svm_model=SVC(gamma='scale')
    svm_model.fit(X, y.values.ravel())
    svm_results = cross_validate(svm_model, X, y.values.ravel(), cv=10, scoring=make_scorer(classification_report_with_accuracy_score))

    print("Average Accuracy: ", svm_results['test_score'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))

    print()
    print(f"========================== Naive Bayes ==========================")

    originalclass = []
    predictedclass = []
    gnb = GaussianNB()
    gnb.fit(X, y.values.ravel())

    nb_results = cross_validate(gnb, X, y.values.ravel(), cv=10, scoring=make_scorer(classification_report_with_accuracy_score))
    # print(len(originalclass))
    # print(len(probas))
    # print(originalclass)
    # probas = cross_val_predict(proba_logreg(), X, y.values.ravel(), cv=10)
    # print(probas)
    # log_loss_score = log_loss(originalclass, probas)
    # print(log_loss_score)
    # print(len(predictedclass))
    print("Average Accuracy: ", nb_results['test_score'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))
