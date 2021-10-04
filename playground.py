from GASTAEN import GASTAEN
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def main():

    models = pickle.load(open('test_models.sav', 'rb'))
    gastaen = GASTAEN(
        models=[models['knn'], models['svm'], models['dt']],
        n_classes=2
    )

    [X_train, y_train, X_test, y_test] = models['X_train'], models['y_train'], models['X_test'], models['y_test']

    # [
    #     [
    #         [0, 0.607, 0.607, 0.607]
    #     ],
    #     [
    #         [0.303, 0.303, 0, 0.303]
    #     ],
    #     [
    #         [0.09, 0.09, 0, 0.09]
    #     ]
    # ]

    # model
    [
        # class
        [
            #pred
            [0, 0.607, 0.607, 0.607],
            [0.607, 0 , 0, 0],
        ]
    ]


    print('Models')
    print(gastaen.get_models())
    print('Weights')
    print(gastaen.get_weights())
    print(sum(gastaen.get_weights()))
    # print(X_test.loc[173])
    print(X_test.head())
    pred = gastaen.predict(X_test)

    print(pred)
    print(y_test)


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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.012, random_state=1)

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

# save_models()

main()