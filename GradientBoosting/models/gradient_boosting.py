import numpy as np

from models.decision_tree import DT
from utils.enums import TaskType


class GradientBoosting:
    def __init__(self, nb_of_weak_learners: int, weight_of_weak_learners: float):
        self.nb_of_weak_learners = nb_of_weak_learners
        self.weight_of_weak_learners = weight_of_weak_learners
        self.zero_weak_learner = None
        self.__weak_learners = []

    def train(self, inputs: np.ndarray, targets: np.ndarray):
        y = self.__init_zero_learner(targets)
        for i in range(1, self.nb_of_weak_learners):
            residuals = targets - y
            decision_stump = DT(task_type=TaskType.regression,
                                max_depth=1,
                                min_entropy=0.01,
                                min_number_of_elem=1)
            decision_stump.train(inputs=inputs, targets=residuals)
            self.__weak_learners.append(decision_stump)
            predictions = decision_stump.get_predictions(inputs)
            y = y + self.weight_of_weak_learners * predictions

    def __init_zero_learner(self, targets: np.ndarray) -> np.ndarray:
        self.zero_weak_learner = np.mean(targets)
        return np.full(shape=len(targets), fill_value=self.zero_weak_learner)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        if len(self.__weak_learners) == 0:
            raise Exception('The model is not trained! Before calling this method, call the train method')
        result = np.full(shape=inputs.shape[0], fill_value=self.zero_weak_learner)
        for i in range(1, self.nb_of_weak_learners):
            predictions = self.__weak_learners[i - 1].get_predictions(inputs)
            result += (predictions * self.weight_of_weak_learners)
        return result
