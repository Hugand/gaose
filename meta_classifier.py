import numpy as np

class MetaClassifier:
    def __init__(self, n_classes, prediction_type='mean'):
        self.n_classes = n_classes
        self.prediction_type = prediction_type

    def predict(self, X, weights):
        weighted_prediction = np.array(weights) * np.array(X).transpose()
        prediction_methods = {
            'mean': self.__mean_prediction,
        }

        return self.__mean_prediction(weighted_prediction, weights)

    def __mean_prediction(self, weighted_prediction, weights):
        predictions = sum(weighted_prediction.transpose()) #* 1/sum(weights)

        return np.round(predictions).transpose() - 1