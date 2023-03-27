from typing import Union

import numpy as np
from easydict import EasyDict


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)

    def weights_init_normal(self, sigma):
        # TODO init weights with values from normal distribution
        pass

    def weights_init_uniform(self, epsilon):
        # TODO init weights with values from uniform distribution BONUS TASK
        pass

    def weights_init_xavier(self, n_in, n_out):
        # TODO Xavier weights initialisation BONUS TASK
        pass

    def weights_init_he(self, n_in):
        # TODO He weights initialisation BONUS TASK
        pass

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        # TODO softmax function realisation
        #  subtract max value of the model_output for numerical stability

        pass

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model confidence (y in lecture)
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        # TODO calculate model output (z in lecture) using matrix multiplication DONT USE LOOPS
        pass

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # TODO calculate gradient for w
        #  slide 10 in presentation
        pass

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # TODO calculate gradient for b
        #  slide 10 in presentation
        pass

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        # TODO update model weights
        #  slide 8, item 2 in presentation for updating weights
        pass

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):
        # TODO one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #
        #  update weights
        #   you can add some other steps if you need
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """
        pass

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # TODO loop stopping criteria - number of iterations of gradient_descent
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        # TODO gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        #  BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid,
                                                                                targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        # TODO target function value calculation
        #  use formula from slide 6 for computational stability
        pass

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        # TODO metrics calculation: accuracy, confusion matrix
        pass

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions
