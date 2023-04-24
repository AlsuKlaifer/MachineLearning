import numpy as np


class RandomForest():

    def __init__(self, nb_trees, max_depth, min_entropy, min_elem, max_nb_dim_to_check, max_nb_thresholds):
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.max_nb_dim_to_check = max_nb_dim_to_check
        self.max_nb_thresholds = max_nb_thresholds

    def train(self, inputs, targets, nb_classes):
        self.trees = []
        for i in range(self.nb_trees):
            pass

    def get_prediction(self, inputs):
        """
        :param inputs: вектора характеристик
        :return: предсказания классов или вектора уверенности
        np.argmax - для предсказания класса
        """
        pass


