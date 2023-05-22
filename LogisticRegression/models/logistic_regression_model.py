import math
from typing import Union

import numpy as np
from easydict import EasyDict

import utils.metrics as metrics
from datasets.base_dataset_classes import BaseClassificationDataset


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        self.weights = None
        self.b = None
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)
        self.data_for_plots = {'epochs': [],
                               'target_function_value_train': [],
                               'accuracy_train': [],
                               'accuracy_valid': []}

    def weights_init_normal(self, sigma):
        # init weights with values from normal distribution
        self.weights = 0.0 + sigma * np.random.randn(self.k, self.d)
        self.b = 0.0 + sigma * np.random.randn(self.k, 1)

    def weights_init_uniform(self, epsilon):
        # init weights with values from uniform distribution BONUS TASK
        self.weights = np.random.uniform(0, epsilon, size=(self.k, self.d))
        self.b = np.random.uniform(0, epsilon, size=(self.k, 1))

    def weights_init_xavier(self, n_in, n_out):
        # Xavier weights initialisation BONUS TASK
        limit = math.sqrt(6.0 / (n_in + n_out))
        self.weights = np.random.uniform(-limit, limit, size=(self.k, self.d))
        self.b = np.random.uniform(-limit, limit, size=(self.k, 1))

    def weights_init_he(self, n_in):
        # He weights initialisation BONUS TASK
        std = math.sqrt(2.0 / n_in)
        self.weights = 0.0 + std * np.random.randn(self.k, self.d)
        self.b = 0.0 + std * np.random.randn(self.k, 1)

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        z = model_output
        z = z - np.max(z)
        y = np.exp(z) / np.sum(np.exp(z), axis=0)
        return y

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model confidence (y in lecture)
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        # calculate model output (z) using matrix multiplication
        z = np.dot(self.weights, inputs.T) + self.b  # w - k x d, inputs - n x d, b - k x 1
        return z

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        y = model_confidence
        gradient_w = np.dot(y - targets.T, inputs)
        # BONUS TASK
        gradient_w_with_reg = gradient_w + self.cfg.reg_coefficient * self.weights
        return gradient_w_with_reg

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        y = model_confidence
        u = np.array([np.ones(targets.shape[0])])  # u - 1 x n
        return np.dot(y - targets.T, u.T)

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        # update model weights
        grad_w = self.__get_gradient_w(inputs, targets, model_confidence)
        grad_b = self.__get_gradient_b(targets, model_confidence)
        self.weights = self.weights - self.cfg.gamma * grad_w
        self.b = self.b - self.cfg.gamma * grad_b

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):
        # one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #  update weights;
        targets_train_onehot = BaseClassificationDataset.onehotencoding(targets_train, self.k)
        y = self.get_model_confidence(inputs_train)
        target_function_value = self.__target_function_value(inputs_train, targets_train_onehot, y)
        self.__validate(inputs_train, targets_train, y)
        self.data_for_plots['epochs'].append(epoch)
        self.data_for_plots['target_function_value_train'].append(target_function_value)
        self.data_for_plots['accuracy_train'].append(self.accuracy)
        accuracy_train = self.accuracy
        confusion_matrix_train = self.confusion_matrix
        if inputs_valid is not None and targets_valid is not None:
            self.__validate(inputs_valid, targets_valid)
            self.data_for_plots['accuracy_valid'].append(self.accuracy)
        if epoch % 10 == 0:
            print(f'Target function value on train set: {target_function_value}')
            print('Confusion matrix on train set:')
            print(confusion_matrix_train)
            print(f'Accuracy on train set: {accuracy_train}')
            if inputs_valid is not None and targets_valid is not None:
                print('Confusion matrix on validation set:')
                print(self.confusion_matrix)
                print(f'Accuracy on validation set: {self.accuracy}')
            print()
        self.__weights_update(inputs_train, targets_train_onehot, y)

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        for epoch in range(self.cfg.nb_epoch):
            print(f'epoch = {epoch}')
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        # gradient_descent with gradient norm stopping criteria BONUS TASK
        epoch = 0
        targets_train_onehot = BaseClassificationDataset.onehotencoding(targets_train, self.k)
        while True:
            y = self.get_model_confidence(inputs_train)
            gradient_b = self.__get_gradient_b(targets_train_onehot, y)
            gradient_w = self.__get_gradient_w(inputs_train, targets_train_onehot, y)
            gradient = np.append(gradient_w, gradient_b)
            gradient_norm = np.linalg.norm(gradient)
            print(f'eposh = {epoch}, gradient_norm = {gradient_norm}')
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            if gradient_norm < self.cfg.gradient_norm_threshold:
                break
            epoch += 1

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        # gradient_descent with stopping criteria - norm of difference between w_k-1 and w_k￼BONUS TASK
        epoch = 0
        w_prev = self.weights
        while True:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            w_curr = self.weights
            difference_norm = np.linalg.norm(w_curr - w_prev)
            epoch += 1
            print(f'eposh = {epoch}, difference_norm = {difference_norm}')
            if difference_norm < self.cfg.difference_norm_threshold:
                break
            w_prev = w_curr

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        # gradient_descent with stopping criteria - metric (accuracy, f1 score or other)
        # value on validation set is not growing BONUS TASK
        epoch = 0
        count = 0  # number of epochs when the accuracy value on the validation dataset did not change
        self.__validate(inputs_valid, targets_valid)
        accuracy_prev = self.accuracy
        print(f'eposh = {epoch}, accuracy_valid = {accuracy_prev} ({count})')
        while True:
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)
            self.__validate(inputs_valid, targets_valid)
            accuracy_curr = self.accuracy
            count = count + 1 if accuracy_curr == accuracy_prev else 0
            epoch += 1
            print(f'eposh = {epoch}, accuracy_valid = {accuracy_curr} ({count})')
            if count >= self.cfg.nb_repeats:
                break
            accuracy_prev = accuracy_curr

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid,
                                                                                targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                model_confidence: Union[np.ndarray, None] = None) -> float:
        # target function value calculation
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        value = 0.0
        for i in range(inputs.shape[0]):
            for k in range(self.k):
                value += (targets[i][k] * (np.log(np.sum(np.exp(model_confidence), axis=0))[i]
                                           - model_confidence[k][i]))
        # BONUS TASK
        value += (np.sum(np.square(self.weights)) * self.cfg.reg_coefficient / 2)
        return value

    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        # metrics calculation: accuracy, confusion matrix
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        self.confusion_matrix = metrics.confusion_matrix(predictions, targets)
        self.accuracy = metrics.accuracy(self.confusion_matrix)

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions

    def batch_gradient_descent(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # gradient is calculated only on a part of data with fix size
        # BONUS TASK
        number_of_bathes = inputs_train.shape[0] // self.cfg.batch_size
        split_intervals = []
        for i in range(1, number_of_bathes + 1):
            split_intervals.append(int(self.cfg.batch_size * i))
        for epoch in range(self.cfg.nb_epoch):
            randomize = np.arange(inputs_train.shape[0])
            np.random.shuffle(randomize)
            inputs_train = inputs_train[randomize]
            targets_train = targets_train[randomize]
            batches_input = np.split(inputs_train, split_intervals)
            batches_target = np.split(targets_train, split_intervals)
            for i in range(number_of_bathes):
                print(f'epoch = {epoch}, batch = {i}')
                targets_train_onehot = BaseClassificationDataset.onehotencoding(batches_target[i], self.k)
                y = self.get_model_confidence(batches_input[i])
                self.__weights_update(batches_input[i], targets_train_onehot, y)
                if epoch == self.cfg.nb_epoch - 1 and i == number_of_bathes - 1:
                    target_function_value = self.__target_function_value(batches_input[i], targets_train_onehot, y)
                    self.__validate(batches_input[i], batches_target[i], y)
                    print(f'Target function value on train set: {target_function_value}')
                    print('Confusion matrix on train set:')
                    print(self.confusion_matrix)
                    print(f'Accuracy on train set: {self.accuracy}')
                    if inputs_valid is not None and targets_valid is not None:
                        self.__validate(inputs_valid, targets_valid)
                        print('Confusion matrix on validation set:')
                        print(self.confusion_matrix)
                        print(f'Accuracy on validation set: {self.accuracy}')
