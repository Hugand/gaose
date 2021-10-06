from random import random, uniform
import numpy as np, numpy.random
from sklearn.metrics import confusion_matrix, f1_score
from weight_change_functions import WeightChangeFunction
from copy import deepcopy

class GAOSTAEN:
    def __init__(self, models=[], n_classes=1, pop_size=100, learning_rate=0.3, max_epochs=1000, weight_change_function='linear'):
        self.weights = []
        self.learning_rate = learning_rate
        self.set_models(models)
        self.pop = []
        self.pop_size = pop_size
        self.error = 0.0
        # self.weight_change = 0.0
        self.weight_change_function = weight_change_function
        self.best_fit = []
        self.best_fit_size = 0.05
        self.f1_scr = 0.0
        self.epochs = 0
        self.max_epochs = max_epochs
        self.n_classes = n_classes
        self.tournment_size = 3
        return

    # Public

    def print_pop(self):
        fits = []
        for p in self.pop:
            fits.append(p['fit'])

        print(fits)

    def fit(self, X, y):
        self.__generate_new_pop(X, y)
        epochs = 0

        while epochs < self.max_epochs:
            print('Epoch: ' + str(epochs))
            self.pop.sort(key = lambda p : p['fit'], reverse=True)
            self.best_fit = deepcopy(
                self.pop[:round(self.pop_size * self.best_fit_size)])
            for h in range(self.pop_size):
                element = self.pop[h]

                # Evaluate weights
                prediction = self.__predict_custom_weights(X, element['weights'])
                prediction_error = self.__calc_error(y, prediction)
                #print(prediction_error)
                weight_change = self.__calc_weight_change(prediction_error)

                # Mutate
                mutated_weight = self.__mutate(element['weights'], weight_change)
                new_prediction = self.__predict_custom_weights(X, element['weights'])
                self.pop[h] = {
                    'weights': mutated_weight,
                    'fit': f1_score(y, new_prediction)
                }

            selected_pool = self.__select_pop(self.tournment_size)

            self.pop = self.best_fit + selected_pool

            if self.pop[0]['fit'] == 1: break

            epochs += 1

        self.pop.sort(key = lambda p : p['fit'], reverse=True)
        self.print_pop()

        best = self.pop[0]
        print(best)
        
        self.weights = best['weights']

        return
    
    def predict(self, X):
        WE = self.__build_weight_encoded_matrix(X, self.weights)
        sumed_we = sum(WE)

        # Get final prediction
        Yx = sumed_we.transpose()
        y = []

        for yx in Yx:
            y.append(np.argmax(yx))

        return y


    def get_weights(self):
        return self.weights
    
    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models
        self.weights = self.__generate_new_weights()

    # Private
    def __predict_custom_weights(self, X, weights):
        WE = self.__build_weight_encoded_matrix(X, weights)
        sumed_we = sum(WE)

        # Get final prediction
        Yx = sumed_we.transpose()
        y = []

        for yx in Yx:
            y.append(np.argmax(yx))

        return y

    def __build_weight_encoded_matrix(self, X, weights):
        WE = []

        for i in range(len(self.models)):
            model = self.models[i]
            prediction = model.predict(X)
            WE_model_prediction = self.__weight_encode_model_prediction(prediction, weights[i])
            WE.append(WE_model_prediction)

        return np.array(WE)

    def __weight_encode_model_prediction(self, prediction, weight):
        we_prediction = []  

        for curr_class in range(self.n_classes):
            we_curr_class = []
            for pred in prediction:
                if pred == curr_class:
                    we_curr_class.append(weight)
                else:
                    we_curr_class.append(0)

            we_prediction.append(np.array(we_curr_class))

        return we_prediction

    def __generate_new_weights(self):
        n_models = len(self.models)

        weights = []
        new_weights = np.random.dirichlet(np.ones(n_models),size=1)[0]

        for w in new_weights:
            weights.append(round(w, 3))

        return weights
        
    def __calc_error(self, y, prediction):
        return 1 - f1_score(y, prediction)

    def __calc_weight_change(self, error):
        wc_function = {
            'linear': WeightChangeFunction.linear_weight_change,
            'quadratic': WeightChangeFunction.quadratic_weight_change,
        }

        return round((wc_function[self.weight_change_function])(error, self.learning_rate), 3)

    def __generate_new_pop(self, X, y):
        self.pop = []

        for i in range(self.pop_size):
            new_weights = self.__generate_new_weights()
            prediction = self.__predict_custom_weights(X, new_weights)

            self.pop.append({
                'weights': new_weights,
                'fit': f1_score(y, prediction)
            })

    def __select_pop(self, tournment_size):
        pool = []

        for element in self.pop:
            best = element

            for i in range(tournment_size):
                random_pos = round(uniform(0, self.pop_size - 1))
                if self.pop[random_pos]['fit'] > best['fit']:
                    best = self.pop[random_pos]
            
            pool.append(best)

        return pool

    def __mutate(self, weights, weight_change):
        random_weight_pos = round(uniform(0, len(weights) - 1))
        sign = ['+', '-'][round(random())]

        random_weight_compensator_pos = random_weight_pos

        while random_weight_compensator_pos == random_weight_pos:
            random_weight_compensator_pos = round(uniform(0, len(weights) - 1))

        if sign == '+':
            weights[random_weight_pos] += weight_change
            weights[random_weight_compensator_pos] -= weight_change
        else:
            weights[random_weight_pos] -= weight_change
            weights[random_weight_compensator_pos] += weight_change

        weights[random_weight_pos] = round(weights[random_weight_pos], 3)
        weights[random_weight_compensator_pos] = round(weights[random_weight_compensator_pos], 3)
        

        return weights
