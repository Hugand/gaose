from random import random, uniform
import numpy as np, numpy.random

class GASTAEN:
    def __init__(self, models=[], n_classes=1):
        self.weights = []
        self.set_models(models)
        self.pop = []
        self.error = 0.0
        self.weight_change = 0.0
        self.best_fit = []
        self.f1_scr = 0.0
        self.epochs = 0
        self.max_epochs = 0
        self.n_classes = n_classes
        return

    # Public

    def fit(self, X, y):
        return
    
    def predict(self, X):
        WE = self.__build_weight_encoded_matrix(X)
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
        self.__generate_new_weights()

    # Private
    def __build_weight_encoded_matrix(self, X):
        WE = []

        for i in range(len(self.models)):
            model = self.models[i]
            prediction = model.predict(X)
            WE_model_prediction = self.__weight_encode_model_prediction(prediction, i)
            WE.append(WE_model_prediction)

        return np.array(WE)

    def __weight_encode_model_prediction(self, prediction, model_index):
        we_prediction = []  

        for curr_class in range(self.n_classes):
            we_curr_class = []
            for pred in prediction:
                if pred == curr_class:
                    we_curr_class.append(self.weights[model_index])
                else:
                    we_curr_class.append(0)


            we_prediction.append(np.array(we_curr_class))

        return we_prediction

    def __generate_new_weights(self):
        n_models = len(self.models)

        self.weights = []
        new_weights = np.random.dirichlet(np.ones(n_models),size=1)[0]

        for w in new_weights:
            self.weights.append(round(w, 3))


    def __calc_error(self):
        return

    def __calc_weight_change(self):
        return

    def __save_best_fit(self):
        return

    def __generate_new_pop(self):
        return

    def __resture_best_fit(self):
        return

    def __select_pop(self):
        return

    def __mutate(self):
        return

    def __calc_f1_score(self):
        return