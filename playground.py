from STENS import STENS
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss, classification_report
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy

def main():

    #_models = pickle.load(open('test_models.sav', 'rb'))
    # models = pickle.load(open('models.sav', 'rb'))
    #data = pickle.load(open('dataset.sav', 'rb'))
    data = pd.read_csv('dataset.csv')
    # data = pd.read_csv('../datasets/winequality-red.csv', delimiter=';')
    y_label = 'DEATH_EVENT' # 'quality'

    X = data.drop(columns=y_label)
    y = data[y_label]

    print(data.describe())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # model = KNeighborsClassifier(n_neighbors=11, weights='distance', p=1)
    # model_cpy = deepcopy(model)

    # model_cpy.fit(X_train, y_train)
    # print(accuracy_score(y_test, model_cpy.predict(X_test)))
    # print(accuracy_score(y_test, model.predict(X_test)))



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
        X_test, y_test,
        models=pipeline_models,
        n_classes=2,
        weight_change_function='quadratic',
        pop_size=30,
        max_epochs=300
    )

    # print(preds)

    stens.fit(X_train, y_train)

    pred = stens.predict(X_test)
    pred2 = stens.predict(X_train)

    print("FINAL train: " + str(accuracy_score(y_train, pred2)))
    print("FINAL: " + str(accuracy_score(y_test, pred)))
    
    
    print('\nTest set:')
    stens.print_weak_learners_performance(X_test, y_test)
    print(stens.weights)
    # print('rf: ' + str(f1_score(y_test, pipeline_models[0].predict(X_test))))
    # print('knn: ' + str(f1_score(y_test, pipeline_models[1].predict(X_test))))
    # print('svm: ' + str(f1_score(y_test, pipeline_models[2].predict(X_test))))
    # print('nb: ' + str(f1_score(y_test, pipeline_models[3].predict(X_test))))

    
    # print(list(pred))
    # print(confusion_matrix(y_test, pred))

def save_models():
    knn = make_pipeline(
        StandardScaler(), KNeighborsClassifier())
    svm = make_pipeline(
        StandardScaler(), SVC())
    dt = make_pipeline(
        StandardScaler(), DecisionTreeClassifier())

    dataset = pd.read_csv('../datasets/heart_failure_dataset.csv')

    X = dataset.drop(columns='DEATH_EVENT')
    y = dataset.DEATH_EVENT

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    knn.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    models = {
        'knn': knn,
        'svm': svm,
        'dt': dt,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

    pickle.dump(models, open('test_models.sav', 'wb'))

#save_models()

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):
        trans = X[self.columns]
        return trans

    def fit(self, X, y=None, **fit_params):
        return self
    

main()

