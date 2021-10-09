from random import random, uniform
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

class STENS:
    def __init__(self, models=[], n_classes=1, pop_size=100, learning_rate=0.8, max_epochs=1000, weight_change_function='linear'):
        self.weights = []
        self.learning_rate = learning_rate
        self.set_models(models)
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.weight_change_function = weight_change_function

        self.meta_model_gnb = GaussianNB()
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
        epochs = 0

        # Make initial prediction
        WL_predictions = []
        for m in self.models:
            WL_predictions.append(m.predict(X))

        transposed_WL_predictions = np.array(WL_predictions).transpose()

        weights = self.__generate_new_weights()
        curr_fit = f1_score(y, self.__fit_predict(weights, transposed_WL_predictions))
        N_neighbors = 100

        # Optimize weights
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
                elif new_weights[chosen_weight] > 1.0: new_weights[chosen_weight] = 1.0

                new_pred = self.__fit_predict(new_weights, transposed_WL_predictions)
                neighbors.append({
                    'weights': new_weights,
                    'fit': f1_score(y, new_pred),
                    'wc': weight_change
                })

            neighbors.sort(key=lambda x: x['fit'], reverse=True)
            st = ''
            for n in neighbors:
                st += '   ' + str(n['weights']) + ' - ' + str(n['wc'])
            
            if neighbors[0]['fit'] >= curr_fit:
                weights = neighbors[0]['weights']
                curr_fit = neighbors[0]['fit']

            print(str(weights) + ' - ' + str(curr_fit))

            epochs += 1

        self.weights = weights
    
    def predict(self, X):
        predictions = []
         # Make initial prediction
        WL_predictions = []
        for m in self.models:
            WL_predictions.append(m.predict(X))

        transposed_WL_predictions = np.array(WL_predictions).transpose()

        return self.__fit_predict(self.weights, transposed_WL_predictions)

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models

    # Private
    def __fit_predict(self, weights, transposed_WL_predictions):
        # Create Weighted Weak Learner's predictions
        wwl = []
        for p in transposed_WL_predictions:
            prediction_batch = []
            for i in range(len(p)):
                prediction_batch.append(weights[i] * p[i])

            wwl.append(prediction_batch)

        final_prediction = sum(np.array(wwl).transpose())

        for i in range(len(final_prediction)):
            total_sum = sum(transposed_WL_predictions[i])
            if total_sum == 0: final_prediction[i] = 0.0
            else: final_prediction[i] = round(final_prediction[i] / total_sum)


        return final_prediction

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
