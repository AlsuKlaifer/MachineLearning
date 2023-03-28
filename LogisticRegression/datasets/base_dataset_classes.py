from abc import ABC, abstractmethod

import numpy as np


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.targets_test = None
        self.targets_valid = None
        self.targets_train = None
        self.inputs_test = None
        self.inputs_valid = None
        self.inputs_train = None
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def divide_into_sets(self):
        # перемешивание
        randomize = np.arange(self.inputs.shape[0])
        np.random.shuffle(randomize)
        inputs = self.inputs[randomize]
        targets = self.targets[randomize]
        # разделение
        index_1 = int(inputs.shape[0] * self.train_set_percent)
        index_2 = int(inputs.shape[0] * (self.train_set_percent + self.valid_set_percent))
        self.inputs_train, self.inputs_valid, self.inputs_test = np.split(inputs, [index_1, index_2])
        self.targets_train, self.targets_valid, self.targets_test = np.split(targets, [index_1, index_2])

    def normalization(self):
        # write normalization method BONUS TASK
        min_input = np.min(self.inputs, axis=0)  # axis - calculate by each dimension
        max_input = np.max(self.inputs, axis=0)  # axis 0 - ищем максимум в столбце
        self.inputs_train = (self.inputs_train - min_input) / (max_input - min_input)
        self.inputs_valid = (self.inputs_valid - min_input) / (max_input - min_input)
        self.inputs_test = (self.inputs_test - min_input) / (max_input - min_input)

    def __get_data_stats(self):
        # calculate mean and std of inputs vectors of training set by each dimension
        mean = np.mean(self.inputs_train, axis=0)
        std = np.std(self.inputs_train, axis=0)
        std[std == 0] = 1
        return mean, std

    def standardization(self):
        # write standardization method (use stats from __get_data_stats)
        mean, std = self.__get_data_stats()
        self.inputs_train = (self.inputs_train - mean) / std
        self.inputs_valid = (self.inputs_valid - mean) / std
        self.inputs_test = (self.inputs_test - mean) / std


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # create matrix of onehot encoding vectors for input targets
        targets = np.array(targets).reshape(-1)
        return np.eye(number_classes)[targets]
