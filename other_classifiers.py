import argparse
import pandas as pd

from sklearn.metrics import make_scorer, accuracy_score, log_loss, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--filename', "-f", default="iris", help="Dataset name")



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

def get_log_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

SCORE_MODELS = {
    'accuracy': make_scorer(classification_report_with_accuracy_score),
    'log_loss': make_scorer(log_loss,  needs_proba=True)
}

if __name__ == "__main__":

    args = parser.parse_args()
    print('Dataset: ', args.filename)
    X, y = preprocess_data(args.filename)

    cv = 3 if args.filename == 'glass' else 10
    print(f"========================== Decision Tree ==========================")
    originalclass = []
    predictedclass = []
    dectree = DecisionTreeClassifier(random_state=2)
    dectree.fit(X, y.values.ravel())
    dectree_results = cross_validate(dectree, X, y.values.ravel(), cv=cv, scoring=SCORE_MODELS)
    print("Average Accuracy: ", dectree_results['test_accuracy'].mean())
    print("Average Log Loss: ", dectree_results['test_log_loss'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))



    print(f"========================== Random Forest ==========================")
    originalclass = []
    predictedclass = []
    rforest = RandomForestClassifier(random_state=2)
    rforest.fit(X, y.values.ravel())
    rf_results = cross_validate(rforest, X, y.values.ravel(), cv=cv, scoring=SCORE_MODELS)
    print("Average Accuracy: ", rf_results['test_accuracy'].mean())
    print("Average Log Loss: ", rf_results['test_log_loss'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))

    print()
    print(f"========================== SVM ==========================")

    originalclass = []
    predictedclass = []
    svm_model=SVC(gamma='scale', probability=True)
    svm_model.fit(X, y.values.ravel())
    svm_results = cross_validate(svm_model, X, y.values.ravel(), cv=cv, scoring=SCORE_MODELS)

    print("Average Accuracy: ", svm_results['test_accuracy'].mean())
    print("Average Log Loss: ", svm_results['test_log_loss'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))

    print()
    print(f"========================== Naive Bayes ==========================")

    originalclass = []
    predictedclass = []
    gnb = GaussianNB()
    gnb.fit(X, y.values.ravel())

    nb_results = cross_validate(gnb, X, y.values.ravel(), cv=cv, scoring=SCORE_MODELS)
    print("Average Accuracy: ", nb_results['test_accuracy'].mean())
    print("Average Log Loss: ", nb_results['test_log_loss'].mean())
    print('Classification report: ')
    print(classification_report(originalclass, predictedclass))
