from random import random, uniform
from typing import final
import numpy as np, numpy.random
from sklearn.metrics import confusion_matrix, f1_score
from weight_change_functions import WeightChangeFunction
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from math import floor
from scipy import stats

class STENS:
    def __init__(self, X, y, models=[], n_classes=1, pop_size=100, learning_rate=0.4, max_epochs=1000, weight_change_function='linear'):
        self.learning_rate = learning_rate
        self.set_models(models)
        # self.weights = self.__generate_new_weights()
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.weight_change_function = weight_change_function

        self.weights = []

        for m in self.models:
            self.weights.append(f1_score(y, m.predict(X)))
            # self.weights.append(1 + f1_score(y, m.predict(X)))

        

        # self.meta_model_gnb = GaussianNB()
        self.meta_model_mlp = MLPClassifier(solver='lbfgs', alpha=0.001, 
            hidden_layer_sizes=(15, 10), random_state=1, learning_rate='adaptive')
        return

    # Public
    def print_pop(self):
        fits = []
        for p in self.pop:
            fits.append(p['fit'])

        print(fits)

    def fit(self, X, y):
        transposed_WL_predictions = self.__make_wl_prediction(X)

        # Optimize weights
        self.weights = self.__optimize(transposed_WL_predictions, y)
    
    def predict(self, X):
        transposed_WL_predictions = self.__make_wl_prediction(X)

        return self.__fit_predict(self.weights, transposed_WL_predictions)

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models

    # Private
    def __fit_predict(self, weights, transposed_WL_predictions):
        # return self.__mean_prediction(transposed_WL_predictions, weights)
        # return self.__floor_argmax_prediction(transposed_WL_predictions, weights)
        return self.__simple_argmax(transposed_WL_predictions, weights)

        

    def __get_wl_f1scores(self, y_true, y_pred):
        wl_f1scores = []

        for i in range(len(self.models)):
            wl_f1scores.append(round(f1_score(y_true, y_pred[i]), 1))

        return wl_f1scores

    def __generate_new_weights(self):
        n_models = len(self.models)

        weights = []
        new_weights = np.random.dirichlet(np.ones(n_models),size=1)[0]

        for w in new_weights:
            weights.append(w)

        return weights
        
    def __calc_error(self, y, prediction):
        return 1 - f1_score(y, prediction)

    def __calc_weight_change(self, error):
        wc_function = {
            'linear': WeightChangeFunction.linear_weight_change,
            'quadratic': WeightChangeFunction.quadratic_weight_change,
        }

        return (wc_function[self.weight_change_function])(error, self.learning_rate)

    def __make_wl_prediction(self, X):
        WL_predictions = []
        for m in self.models:
            WL_predictions.append(m.predict(X))

        return np.array(WL_predictions).transpose()

    def __optimize(self, transposed_WL_predictions, y):
        weights = self.weights
        curr_fit = f1_score(y, self.__fit_predict(weights, transposed_WL_predictions))
        N_neighbors = 100
        epochs = 0

        while epochs < self.max_epochs:
            error = 1 - curr_fit

            neighbor_distance = self.__calc_weight_change(error)
            neighbors = []
            
            for i in range(N_neighbors):
                weight_change = neighbor_distance * [1, -1][round(random())]
                chosen_weight = round(uniform(0, len(weights)-1))

                new_weights = deepcopy(weights)
                new_weights[chosen_weight] = new_weights[chosen_weight] + weight_change
                if new_weights[chosen_weight] < 0.0: new_weights[chosen_weight] = 0.0
                # elif new_weights[chosen_weight] > 1.0: new_weights[chosen_weight] = 1.0

                new_pred = self.__fit_predict(new_weights, transposed_WL_predictions)
                neighbors.append({
                    'weights': new_weights,
                    'fit': f1_score(y, new_pred),
                    'wc': weight_change
                })

            neighbors.sort(key=lambda x: x['fit'], reverse=True)
            
            if neighbors[0]['fit'] >= curr_fit:
                weights = neighbors[0]['weights']
                curr_fit = neighbors[0]['fit']

            print('[' + str(epochs) + '] => ' + str(weights) + ' - ' + str(curr_fit))

            epochs += 1

        return weights