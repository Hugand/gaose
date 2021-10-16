from GAOSE import GAOSE
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss, classification_report
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

def main():
    data = pd.read_csv('dataset.csv')
    # data = pd.read_csv('../datasets/winequality-red.csv', delimiter=';')
    # data = pd.read_csv('../datasets/adult.data', delimiter=',')
    y_label = 'DEATH_EVENT'
    # y_label = 'quality'
    # y_label = 'classif'

    # categorical = ['workclass', 'education', 'marital_status', 'sex', 'native_country',
    #     'classif', 'occupation', 'relationship', 'race']

    # encoder = OrdinalEncoder()
    # mf_imputer = SimpleImputer(strategy='most_frequent')
    # mean_imputer = SimpleImputer(strategy='most_frequent')
    # data[categorical] = encoder.fit_transform(data[categorical])
    # data[categorical] = mf_imputer.fit_transform(data[categorical])
    # data[data.columns] = mean_imputer.fit_transform(data[data.columns])

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
            SVC(C=6, kernel='rbf')
        ),
        make_pipeline(
            MinMaxScaler(),
            SVC()
        ),
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
            n_neighbors=5, weights='distance', p=2)
        ),
        make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
            n_neighbors=3, weights='distance', p=1)
        ),
        make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(criterion='gini', max_leaf_nodes=20)),
    ]
 
    gaose = GAOSE(
        models=pipeline_models,
        n_classes=2,
        pInstances=0.4,
        pFeatures=0.3,
        eval_metric='f1-score'
    )

    gaose.fit(
        X_train, y_train,
        pop_size=40,
        max_epochs=3000,
        crossover_type='1pt')

    pred_test = gaose.predict(X_test)
    pred_train = gaose.predict(X_train)

    print("FINAL train: " + str(accuracy_score(y_train, pred_train)))
    print("FINAL test f1: " + str(f1_score(y_train, pred_train, average='weighted')))
    print("FINAL test: " + str(accuracy_score(y_test, pred_test)))
    print("FINAL test f1: " + str(f1_score(y_test, pred_test, average='weighted')))
    print(confusion_matrix(y_test, pred_test))
    
    print('\nTest set:')
    print('Learners perfomance: ')
    gaose.print_weak_learners_performance(X_test, y_test)
    print('Ensemble weights: ')
    print(gaose.weights)


main()

