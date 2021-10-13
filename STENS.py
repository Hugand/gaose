import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from optimizers import hill_climbing_optimizer
from ga_optimizer import GAOptimizer
from meta_classifier import MetaClassifier

class STENS:
    def __init__(self, X, y, models=[], n_classes=1, pop_size=100, learning_rate=0.4, max_epochs=1000, weight_change_function='linear'):
        self.learning_rate = learning_rate
        self.set_models(models)
        self.n_classes = n_classes
        self.max_epochs = max_epochs
        self.weights = None
        self.meta_classifier = MetaClassifier(n_classes, len(self.models))

    # Public
    def print_pop(self):
        fits = []
        for p in self.pop:
            fits.append(p['fit'])

        print(fits)

    def fit(self, X, y):
        n_models = len(self.models)
        X_train, X_mm, y_train, y_mm = train_test_split(X, y, test_size=0.3)
        X_batches = []
        y_batches = []

        # Split training set
        batch_size = round(len(X_train) / n_models)

        for i in range(n_models):
            curr_pos = batch_size*i
            X_batches.append(X_train[curr_pos:curr_pos+batch_size])
            y_batches.append(y_train[curr_pos:curr_pos+batch_size])

        wl_predictions = []

        # Train the weak learners and get their predictions on test set
        for i in range(len(self.models)):
            self.models[i].fit(X_batches[i], y_batches[i])
            wl_predictions.append(self.models[i].predict(X_mm) + 1)

        # Optimize weights
        ga_optimizer = GAOptimizer(
            n_models, pop_size=40, n_generations=3000)
        self.weights = ga_optimizer.optimize(wl_predictions, y_mm, self.meta_classifier)

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

        return self.meta_classifier.predict(wl_predictions, self.weights)

    def get_models(self):
        return self.models

    def set_models(self, models):
        self.models = models