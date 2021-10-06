from GAOSTAEN import GAOSTAEN
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.naive_bayes import GaussianNB

def main():

    _models = pickle.load(open('test_models.sav', 'rb'))
    # models = pickle.load(open('models.sav', 'rb'))
    #data = pickle.load(open('dataset.sav', 'rb'))
    data = pd.read_csv('dataset.csv')
    X = data.drop(columns='DEATH_EVENT')
    y = data.DEATH_EVENT

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    columns_006 = ['age', 'anaemia', 'diabetes', 'ejection_fraction',
        'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
        'sex', 'smoking', 'time']

    columns_003 = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

    columns_03 = ['ejection_fraction', 'serum_creatinine', 'time']

    pipeline_models = [
        make_pipeline(
            DimensionalityReducer(columns_03),
            RandomForestClassifier(
                n_estimators=140, criterion='entropy'
        )),
        make_pipeline(
            DimensionalityReducer(columns_03),
            KNeighborsClassifier(
            n_neighbors=11, weights='distance', p=1
        )),
        make_pipeline(
            DimensionalityReducer(columns_03),
            SVC(C=12, kernel='rbf')
        ),
        make_pipeline(
            DimensionalityReducer(columns_03),
            GaussianNB()
        ),
    ]

    gaostaen = GAOSTAEN(
        models=pipeline_models,
        n_classes=2,
        weight_change_function='quadratic',
        pop_size=50
    )
    # [X_train, y_train, X_test, y_test] = _models['X_train'], _models['y_train'], _models['X_test'], _models['y_test']

    for i in range(len(pipeline_models)):
        pipeline_models[i].fit(X_train, y_train)


    print('Models')
    print(gaostaen.get_models())
    print('Weights')
    print(gaostaen.get_weights())
    print(sum(gaostaen.get_weights()))
    print('')

    print('Train set:')
    print('rf: ' + str(f1_score(y_train, pipeline_models[0].predict(X_train))))
    print('knn: ' + str(f1_score(y_train, pipeline_models[1].predict(X_train))))
    print('svm: ' + str(f1_score(y_train, pipeline_models[2].predict(X_train))))
    print('nb: ' + str(f1_score(y_train, pipeline_models[3].predict(X_train))))
    print('\nTest set:')
    print('rf: ' + str(f1_score(y_test, pipeline_models[0].predict(X_test))))
    print('knn: ' + str(f1_score(y_test, pipeline_models[1].predict(X_test))))
    print('svm: ' + str(f1_score(y_test, pipeline_models[2].predict(X_test))))
    print('nb: ' + str(f1_score(y_test, pipeline_models[3].predict(X_test))))

    # print(X_test.head())
    gaostaen.fit(X_train, y_train)
    pred = gaostaen.predict(X_test)
    pred2 = gaostaen.predict(X_train)

    print("FINAL: " + str(f1_score(y_test, pred)))
    print("FINAL train: " + str(f1_score(y_train, pred2)))
    print(pred)

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
        trans = X
        return trans

    def fit(self, X, y=None, **fit_params):
        return self
    

main()

