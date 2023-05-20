import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_index = None
        self.split_value = None
        self.terminal_node = None


class DT:
    def __init__(self, max_depth: int, min_entropy: float = 0, min_number_of_elem: int = 1):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_number_of_elem = min_number_of_elem
        self.root = Node()

    def train(self, inputs: np.ndarray, targets: np.ndarray, weights: np.ndarray):
        self.__nb_dim = inputs.shape[1]  # сколько компонент во входном векторе
        self.__all_dim = np.arange(self.__nb_dim)  # массив от 0 до self.__nb_dim - 1
        self.__get_axis = self.__get_all_axis
        self.__get_threshold = self.__generate_all_threshold
        self.__unique_targets = np.unique(targets)
        self.__k = len(self.__unique_targets)  # кол-во классов
        entropy_val = self.__shannon_entropy(targets, weights)
        self.__build_tree(inputs, targets, self.root, 0, entropy_val, weights)

    def __get_all_axis(self) -> np.ndarray:
        return self.__all_dim

    def __create_term_arr(self, targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
        term_array = np.zeros(self.__k)
        n = np.sum(weights)
        unique_classes, indices = np.unique(targets, return_inverse=True)
        indices_of_classes = np.arange(self.__k)
        indices_of_classes = indices_of_classes[np.searchsorted(self.__unique_targets, unique_classes,
                                                                sorter=indices_of_classes)]
        term_array[indices_of_classes] = np.bincount(indices, weights=weights) / n
        return term_array

    def __generate_all_threshold(self, inputs: np.ndarray):
        return np.unique(inputs)

    @staticmethod
    def __shannon_entropy(targets: np.ndarray, weights: np.ndarray) -> float:
        unique_classes, indices = np.unique(targets, return_inverse=True)
        # indices - массив длины len(targets) из индексов первых вхождений уникальных элементов массива targets
        n = np.sum(weights)
        n_for_classes = np.bincount(indices, weights=weights)  # считаем суммы вессов для классов
        return -np.sum((n_for_classes / n) * np.log2(n_for_classes / n))

    def __inf_gain(self, targets_left: np.ndarray, targets_right: np.ndarray, node_entropy: float,
                   weights_left: np.ndarray, weights_right: np.ndarray):
        left_entropy = self.__shannon_entropy(targets_left, weights_left)
        right_entropy = self.__shannon_entropy(targets_right, weights_right)
        n_left = np.sum(weights_left)
        n_right = np.sum(weights_right)
        n = n_left + n_right
        inf_gain = node_entropy - (left_entropy * n_left / n) - (right_entropy * n_right / n)
        return inf_gain, left_entropy, right_entropy

    def __build_splitting_node(self, inputs: np.ndarray, targets: np.ndarray, entropy: float, weights: np.ndarray):
        split_index = 0
        split_value = 0
        entropy_left_max = 0
        entropy_right_max = 0
        indices_left_max = None
        indices_right_max = None
        information_gain_max = -1
        for axis in self.__get_axis():
            for threshold in self.__get_threshold(inputs[:, axis]):
                indices_left = inputs[:, axis] <= threshold
                indices_right = inputs[:, axis] > threshold
                information_gain, entropy_left, entropy_right = \
                    self.__inf_gain(targets[indices_left], targets[indices_right], entropy,
                                    weights[indices_left], weights[indices_right])
                if information_gain > information_gain_max:
                    split_index = axis
                    split_value = threshold
                    entropy_left_max = entropy_left
                    entropy_right_max = entropy_right
                    indices_left_max = indices_left
                    indices_right_max = indices_right
                    information_gain_max = information_gain
        return split_index, split_value, indices_left_max, indices_right_max, entropy_left_max, entropy_right_max

    def __build_tree(self, inputs: np.ndarray, targets: np.ndarray, node: Node, depth: int, entropy: float,
                     weights: np.ndarray):
        n = len(targets)
        if depth >= self.max_depth or entropy <= self.min_entropy or n <= self.min_number_of_elem:
            node.terminal_node = self.__create_term_arr(targets, weights)
        else:
            split_index, split_value, indices_left, indices_right, entropy_left, entropy_right = \
                self.__build_splitting_node(inputs, targets, entropy, weights)
            node.split_index = split_index
            node.split_value = split_value
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[indices_left], targets[indices_left], node.left_child, depth + 1, entropy_left,
                              weights[indices_left])
            self.__build_tree(inputs[indices_right], targets[indices_right], node.right_child, depth + 1, entropy_right,
                              weights[indices_right])

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        node = self.root
        results = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            while node.terminal_node is None:
                if inputs[i][node.split_index] > node.split_value:
                    node = node.right_child
                else:
                    node = node.left_child
            results[i] = self.__unique_targets[np.argmax(node.terminal_node)]
            node = self.root
        return results
