from abc import ABC, abstractmethod

import numpy as np


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
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
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework

        pass

    def normalization(self):
        # TODO write normalization method BONUS TASK
        pass

    def get_data_stats(self):
        # TODO calculate mean and std of inputs vectors of training set by each dimension
        pass

    def standartization(self):
        # TODO write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP
        pass


class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # TODO create matrix of onehot encoding vactors for input targets
        #  it is possible to do it without loop
        pass
