import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from optimizers import hill_climbing_optimizer
from ga_optimizer import GAOptimizer
from meta_classifier import MetaClassifier
import pandas as pd
from copy import deepcopy

class STENS:
    def __init__(self, models=[], n_classes=1, pop_size=100,
        learning_rate=0.4, max_epochs=1000, pInstances=1.0, pFeatures=1.0,
        crossover_type='1pt'):
        self.learning_rate = learning_rate
        self.models = models
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.weights = None
        self.meta_classifier = MetaClassifier(n_classes, len(self.models))
        self.pInstances = pInstances
        self.pFeatures = pFeatures
        self.selected_features = []
        self.ga_optimizer = GAOptimizer(
            len(models), pop_size=pop_size, n_generations=max_epochs, crossover_type=crossover_type)

    # Public
    def print_pop(self):
        fits = []
        for p in self.pop:
            fits.append(p['fit'])

        print(fits)

    def __instance_data_sampling(self, X, y):
        df = pd.DataFrame(data=deepcopy(X), columns=X.columns)
        df.insert(len(X.columns), 'y_label', y)

        sampled_df = df.sample(frac=self.pInstances)

        return sampled_df.drop(columns='y_label'), sampled_df.y_label

    def __feature_data_sampling(self, X):
        return X.sample(frac=self.pFeatures, axis='columns')

    def __sample_data(self, X, y):
        X_inst_sampled, y_inst_sampled = self.__instance_data_sampling(X, y)
        X_inst_sampled = self.__feature_data_sampling(X_inst_sampled)

        return X_inst_sampled, y_inst_sampled

    def fit(self, X, y):
        n_models = len(self.models)
        X_train, X_mm, y_train, y_mm = train_test_split(X, y, test_size=0.3)
        wl_predictions = []
        self.selected_features = []

        # Train the weak learners and get their predictions on test set
        for i in range(len(self.models)):
            X_inst_sampled, y_inst_sampled = self.__sample_data(X_train, y_train)
            self.models[i].fit(X_inst_sampled, y_inst_sampled)
            self.selected_features.append(list(X_inst_sampled.columns))
            print(X_mm[self.selected_features[i]].head())
            wl_predictions.append(self.models[i].predict(X_mm[self.selected_features[i]]) + 1)

        # Optimize weights
        self.weights = self.ga_optimizer.optimize(wl_predictions, y_mm, self.meta_classifier)

    def print_weak_learners_performance(self, X, y):
        scores = []
        for i in range(len(self.models)):
            scores.append(accuracy_score(y, self.models[i].predict(X[self.selected_features[i]])))

        print(scores)
    
    def predict(self, X):
        # Train the weak learners and get their predictions on test set
        wl_predictions = []
        for i in range(len(self.models)):
            wl_predictions.append(self.models[i].predict(X[self.selected_features[i]]) + 1)

        return self.meta_classifier.predict(wl_predictions, self.weights)

    def get_models(self):
        return self.models
        