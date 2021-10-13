from STENS import STENS
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss, classification_report
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy

def main():
    data = pd.read_csv('dataset.csv')
    # data = pd.read_csv('../datasets/winequality-red.csv', delimiter=';')
    y_label = 'DEATH_EVENT'
    # y_label = 'quality'

    X = data.drop(columns=y_label)
    y = data[y_label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    pipeline_models = [
        make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(criterion='entropy')),
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
            n_neighbors=11, weights='distance', p=1
        )),
        make_pipeline(
            StandardScaler(),
            SVC(C=12, kernel='rbf')
        ),
        make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        make_pipeline(
            StandardScaler(),
            GaussianNB()
        ),
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
            n_neighbors=5, weights='distance', p=2)
        ),
    ]
 
    stens = STENS(
        models=pipeline_models,
        n_classes=2,
        pop_size=30,
        max_epochs=300,
        pInstances=0.6,
        pFeatures=0.5,
    )

    stens.fit(X_train, y_train)

    pred_test = stens.predict(X_test)
    pred_train = stens.predict(X_train)

    print(pred_test)
    print(y_test)

    print("FINAL train: " + str(accuracy_score(y_train, pred_train)))
    print("FINAL test: " + str(accuracy_score(y_test, pred_test)))
    # print("FINAL test f1: " + str(f1_score(y_test, pred_test)))
    print(confusion_matrix(y_test, pred_test))
    
    print('\nTest set:')
    print('Learners perfomance: ')
    stens.print_weak_learners_performance(X_test, y_test)
    print('Ensemble weights: ')
    print(stens.weights)


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        trans = X[self.columns]
        return trans

    def fit(self, X, y=None, **fit_params):
        return self
    

main()

