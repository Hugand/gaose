import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from ga_optimizer import GAOptimizer
from meta_classifier import MetaClassifier
import pandas as pd
from copy import deepcopy

class GAOSE:
    def __init__(self, models=[], n_classes=1, pInstances=1.0, pFeatures=1.0, eval_metric='accuracy'):
        self.models = models
        self.n_classes = n_classes
        self.weights = None
        self.meta_classifier = MetaClassifier(n_classes, len(self.models))
        self.pInstances = pInstances
        self.pFeatures = pFeatures
        self.selected_features = []
        self.eval_metric = eval_metric
        
    # Public
    def fit(self, X, y,
        holdout_frac=0.3,
        pop_size=30, max_epochs=1000,
        mutation_prob=0.1,
        crossover_prob=0.7, crossover_type='1pt',
        selection_type='tournment', best_fit_frac=0.05, poison_prob=0.3,):

        n_models = len(self.models)
        ga_optimizer = GAOptimizer(
            n_models, pop_size=pop_size,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            n_generations=max_epochs, crossover_type=crossover_type,
            selection_type=selection_type,
            poison_prob=poison_prob, best_fit_frac=best_fit_frac, eval_metric=self.eval_metric)

        X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=holdout_frac)
        wl_predictions = []
        self.selected_features = []

        # Train the weak learners and get their predictions on test set
        for i in range(len(self.models)):
            X_inst_sampled, y_inst_sampled = self.__sample_data(X_train, y_train)
            self.models[i].fit(X_inst_sampled, y_inst_sampled)
            self.selected_features.append(list(X_inst_sampled.columns))
            wl_predictions.append(self.models[i].predict(X_holdout[self.selected_features[i]]) + 1)

        # Optimize weights
        self.weights = ga_optimizer.optimize(wl_predictions, y_holdout, self.meta_classifier)

    def get_weak_learners_performance(self, X, y):
        scores = []
        for i in range(len(self.models)):
            scores.append(self.__eval_performance(y, self.models[i].predict(X[self.selected_features[i]])))

        return scores
    
    def predict(self, X):
        wl_predictions = []
        for i in range(len(self.models)):
            wl_predictions.append(self.models[i].predict(X[self.selected_features[i]]) + 1)

        return self.meta_classifier.predict(wl_predictions, self.weights)

    def get_models(self):
        return self.models

    # Private
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

    def __eval_performance(self, y_true, y_pred):
        if(self.eval_metric == 'accuracy'):
            return accuracy_score(y_true, y_pred)
        elif(self.eval_metric == 'f1-score'):
            return f1_score(y_true, y_pred, average='weighted')