from random import random, uniform
from typing import final
import numpy as np, numpy.random
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from copy import deepcopy
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from math import floor
from scipy import stats
from optimizers import hill_climbing_optimizer
from ga_optimizer import GAOptimizer

class STENS:
    def __init__(self, X, y, models=[], n_classes=1, pop_size=100, learning_rate=0.4, max_epochs=1000, weight_change_function='linear'):
        self.learning_rate = learning_rate
        self.set_models(models)
        # self.weights = self.__generate_new_weights()
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.weight_change_function = weight_change_function

        self.weights = self.__generate_new_weights()

        self.meta_model_mlp = DecisionTreeClassifier()
        #  MLPClassifier(solver='lbfgs', alpha=0.05, max_iter=1000,
        #      hidden_layer_sizes=(10,), activation='relu', random_state=1, learning_rate='adaptive')

        return

    # Public
    def print_pop(self):
        fits = []
        for p in self.pop:
            fits.append(p['fit'])

        print(fits)

    def fit(self, X, y):
        n_models = len(self.models)
        X_train, X_mm, y_train, y_mm = train_test_split(X, y, test_size=0.3)
        X_mm, X_valid, y_mm, y_valid = train_test_split(X_mm, y_mm, test_size=0.35)
        X_batches = []
        y_batches = []

        # Split training set
        batch_size = round(len(X_train) / n_models)

        for i in range(n_models):
            curr_pos = batch_size*i
            X_batches.append(X_train[curr_pos:curr_pos+batch_size])
            y_batches.append(y_train[curr_pos:curr_pos+batch_size])

        wl_predictions = []
        wl_valid_predictions = []

        # Train the weak learners and get their predictions on test set
        for i in range(len(self.models)):
            self.models[i].fit(X_batches[i], y_batches[i])
            wl_predictions.append(self.models[i].predict(X_mm) + 1)
            wl_valid_predictions.append(self.models[i].predict(X_valid) + 1)

        # weighted_wl_predictions = np.array(wl_predictions).transpose() * np.array(self.weights)
        # self.meta_model_mlp.fit(weighted_wl_predictions, y_mm)
        
        # Optimize weights
        # self.weights = hill_climbing_optimizer(
        #     wl_predictions, y_mm, wl_valid_predictions, y_valid,
        #     self.meta_model_mlp, self.weights, self.max_epochs)
        ga_optimizer = GAOptimizer(
            n_models, self.meta_model_mlp, wl_predictions, y_mm, wl_valid_predictions, y_valid,
            pop_size=50)
        self.weights = ga_optimizer.optimize()

            
        weighted_wl_predictions = np.array(wl_predictions).transpose() * np.array(self.weights)
        self.meta_model_mlp.fit(weighted_wl_predictions, y_mm)
        

    def print_weak_learners_performance(self, X, y):
        scores = []
        for i in range(len(self.models)):
            scores.append(accuracy_score(y, self.models[i].predict(X)))

        print(scores)
    
    def predict(self, X):
        # Train the weak learners and get their predictions on test set
        wl_predictions = []
        for i in range(len(self.models)):
            wl_predictions.append(self.models[i].predict(X) + 1)

        weighted_wl_predictions = np.array(wl_predictions).transpose() * np.array(self.weights)

        return self.meta_model_mlp.predict(weighted_wl_predictions)

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models

    # Private
    def __generate_new_weights(self):
        n_models = len(self.models)

        weights = []
        new_weights = np.random.dirichlet(np.ones(n_models),size=1)[0]

        for w in new_weights:
            weights.append(w)

        return weights