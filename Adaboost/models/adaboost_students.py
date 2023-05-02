import math

import numpy as np

from models.decision_tree import DT


class Adaboost:
    def __init__(self, number_of_weak_classifiers: int):
        self.number_of_weak_classifiers = number_of_weak_classifiers
        self.__weak_classifiers = []

    def __init_weights(self, targets: np.ndarray):
        """ initialization of input variables weights """
        # weights = np.full(shape=len(targets), fill_value=1 / len(targets))

        # method from the lecture
        weights = np.zeros(len(targets))
        unique_classes, count = np.unique(targets, return_counts=True)
        k = len(unique_classes)
        for i in range(k):
            weights[targets == unique_classes[i]] = 1 / (k * count[i])
        return weights

    def __update_weights(self, targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray,
                         weak_classifier_weight: float) -> np.ndarray:
        exponents = np.ones(len(targets))
        exponents[predictions != targets] = np.exp(weak_classifier_weight)
        normalization_coeff = np.sum(weights * exponents)
        return weights * exponents / normalization_coeff

    def __calculate_error(self, targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> float:
        """ weak classifier error calculation without loops """
        error = np.sum(weights[predictions != targets])
        return float(error)

    def __calculate_classifier_weight(self, error: float) -> float:
        """ weak classifier weight calculation without loops """
        return np.log((1 - error) / error)

    def train(self, inputs: np.ndarray, targets: np.ndarray):
        weights = self.__init_weights(targets)
        for i in range(self.number_of_weak_classifiers):
            decision_tree = DT(max_depth=1, min_entropy=0.01, min_number_of_elem=1)
            decision_tree.train(inputs, targets, weights)
            predictions = decision_tree.get_predictions(inputs)
            error = self.__calculate_error(targets, predictions, weights)
            alpha = self.__calculate_classifier_weight(error)
            self.__weak_classifiers.append((decision_tree, alpha))
            if math.isclose(error, 0.5):
                self.number_of_weak_classifiers = len(self.__weak_classifiers)
                break
            weights = self.__update_weights(targets, predictions, weights, alpha)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        """ adaboost get predictions """
        if len(self.__weak_classifiers) == 0:
            raise Exception('The model is not trained! Before calling this method, call the train method')
        all_predictions = np.zeros((self.number_of_weak_classifiers, inputs.shape[0]))
        for i in range(self.number_of_weak_classifiers):
            all_predictions[i] = self.__weak_classifiers[i][0].get_predictions(inputs) * self.__weak_classifiers[i][1]
        return np.sign(np.sum(all_predictions, axis=0))
