from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from random import random, uniform
from copy import deepcopy
import numpy as np
from weight_change_functions import WeightChangeFunction

def ga_optimizer():
    return # return function in respective optimizer

def hill_climbing_optimizer(wl_predictions, y, X_valid, y_valid, meta_model, curr_weights, max_epochs):
    weights = curr_weights
    curr_fit = accuracy_score(y_valid, meta_model.predict(
        np.array(X_valid).transpose() * np.array(weights)
    ))
    N_neighbors = 100
    epochs = 0

    while epochs < max_epochs:
        error = 1 - curr_fit

        neighbor_distance = __calc_weight_change(error)
        neighbors = []
        
        for i in range(N_neighbors):
            meta_model_cpy = deepcopy(meta_model)
            # Change weight
            weight_change = random() #neighbor_distance
            chosen_weight = round(uniform(0, len(weights)-1))

            new_weights = deepcopy(weights)
            new_weights[chosen_weight] = new_weights[chosen_weight] + weight_change

            new_weights = __normalize_weights(new_weights)

            # Calc new weak learners predictions
            weighted_predictions = np.array(wl_predictions).transpose() * np.array(new_weights)
            weighted_valid_predictions = np.array(X_valid).transpose() * np.array(new_weights)
            
            # Train and evaluate meta model
            meta_model_cpy.fit(weighted_predictions, y)
            valid_predictions = meta_model_cpy.predict(weighted_valid_predictions)
            
            neighbors.append({
                'weights': new_weights,
                'fit': accuracy_score(y_valid, valid_predictions)
            })

        neighbors.sort(key=lambda x: x['fit'], reverse=True)
        
        if neighbors[0]['fit'] > curr_fit:
            weights = neighbors[0]['weights']
            curr_fit = neighbors[0]['fit']

        print('[' + str(epochs) + '] => ' + str(weights) + ' - ' + str(curr_fit))

        epochs += 1

    return weights

def __normalize_weights(weights):
    total = sum(weights)
    normalized_weights = []

    for w in weights:
        normalized_weights.append(w / total)

    return normalized_weights

def __calc_weight_change(error):
    wc_function = {
        'linear': WeightChangeFunction.linear_weight_change,
        'quadratic': WeightChangeFunction.quadratic_weight_change,
    }

    return WeightChangeFunction.quadratic_weight_change(error, 0.7)

    # return (wc_function[self.weight_change_function])(error, self.learning_rate)
