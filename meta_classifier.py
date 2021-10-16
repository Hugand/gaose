import numpy as np

class MetaClassifier:
    def __init__(self, n_classes, prediction_type='mean'):
        self.n_classes = n_classes
        self.prediction_type = prediction_type

    def predict(self, X, weights):
        weighted_prediction = np.array(weights) * np.array(X).T
        # prediction_methods = {
        #     'mean': self.__mean_prediction,
        # }

        return self.__mean_prediction(weighted_prediction, weights)

    def __mean_prediction(self, weighted_prediction, weights):
        transposed_weighted_predictions = weighted_prediction.T
        predictions = np.dot(
            np.ones((1, len(transposed_weighted_predictions))),
            transposed_weighted_predictions
        )[0]

        return np.round(predictions) - 1