import numpy as np

from utils.enums import TaskType


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_index = None
        self.split_value = None
        self.terminal_node = None


class DT:
    def __init__(self, task_type, max_depth: int, min_entropy: float = 0, min_number_of_elem: int = 1):
        if task_type != TaskType.classification and task_type != TaskType.regression:
            raise Exception('No such task type. Choose classification or regression.')
        self.task_type = task_type
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_number_of_elem = min_number_of_elem
        self.root = Node()
        # self.max_nb_thresholds = max_nb_thresholds

    def train(self, inputs: np.ndarray, targets: np.ndarray):
        self.__nb_dim = inputs.shape[1]  # сколько компонент во входном векторе
        self.__all_dim = np.arange(self.__nb_dim)  # массив от 0 до self.__nb_dim - 1
        self.__get_axis = self.__get_all_axis
        self.__get_threshold = self.__generate_all_threshold
        if self.task_type == TaskType.classification:
            self.__k = len(np.unique(targets))  # кол-во классов
            entropy_val = self.__shannon_entropy(targets, len(targets))
            self.__build_tree(inputs, targets, self.root, 1, entropy_val)
        elif self.task_type == TaskType.regression:
            disp_val = self.__disp(targets)
            self.__build_tree(inputs, targets, self.root, 1, disp_val)

    def __get_random_axis(self):
        pass

    def __get_all_axis(self) -> np.ndarray:
        return self.__all_dim

    def __create_term_arr(self, targets: np.ndarray) -> np.ndarray:
        if self.task_type == TaskType.classification:
            term_array = np.zeros(self.__k)
            unique_classes, counts = np.unique(targets, return_counts=True)
            term_array[unique_classes] = counts / len(targets)
            return term_array
        elif self.task_type == TaskType.regression:
            return np.mean(targets)

    def __generate_all_threshold(self, inputs: np.ndarray):
        return np.sort(inputs, axis=None)

    def __generate_random_threshold(self, inputs):
        pass

    @staticmethod
    def __disp(targets: np.ndarray) -> float:
        return np.var(targets) if len(targets) > 0 else 0

    @staticmethod
    def __shannon_entropy(targets: np.ndarray, n: int) -> float:
        unique_classes, counts = np.unique(targets, return_counts=True)
        return -np.sum((counts / n) * np.log2(counts / n))

    def __inf_gain(self, targets_left: np.ndarray, targets_right: np.ndarray, node_entropy: float, n: int):
        left_entropy = 0
        right_entropy = 0
        if self.task_type == TaskType.classification:
            left_entropy = self.__shannon_entropy(targets_left, n)
            right_entropy = self.__shannon_entropy(targets_right, n)
        elif self.task_type == TaskType.regression:
            left_entropy = self.__disp(targets_left)
            right_entropy = self.__disp(targets_right)
        inf_gain = node_entropy - (left_entropy * len(targets_left) / n) - (right_entropy * len(targets_right) / n)
        return inf_gain, left_entropy, right_entropy

    def __build_splitting_node(self, inputs: np.ndarray, targets: np.ndarray, entropy: float, n):
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
                    self.__inf_gain(targets[indices_left], targets[indices_right], entropy, n)
                if information_gain > information_gain_max:
                    split_index = axis
                    split_value = threshold
                    entropy_left_max = entropy_left
                    entropy_right_max = entropy_right
                    indices_left_max = indices_left
                    indices_right_max = indices_right
                    information_gain_max = information_gain
        return split_index, split_value, indices_left_max, indices_right_max, entropy_left_max, entropy_right_max

    def __build_tree(self, inputs: np.ndarray, targets: np.ndarray, node: Node, depth: int, entropy: float):
        n = len(targets)
        if depth >= self.max_depth or entropy <= self.min_entropy or n <= self.min_number_of_elem:
            node.terminal_node = self.__create_term_arr(targets)
        else:
            split_index, split_value, indices_left, indices_right, entropy_left, entropy_right = \
                self.__build_splitting_node(inputs, targets, entropy, n)
            node.split_index = split_index
            node.split_value = split_value
            node.left_child = Node()
            node.right_child = Node()
            self.__build_tree(inputs[indices_left], targets[indices_left], node.left_child, depth + 1, entropy_left)
            self.__build_tree(inputs[indices_right], targets[indices_right], node.right_child, depth + 1, entropy_right)

    def get_predictions(self, inputs: np.ndarray) -> np.ndarray:
        node = self.root
        results = np.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            while node.terminal_node is None:
                if inputs[i][node.split_index] > node.split_value:
                    node = node.right_child
                else:
                    node = node.left_child
            if self.task_type == TaskType.classification:
                results[i] = np.argmax(node.terminal_node)
            elif self.task_type == TaskType.regression:
                results[i] = node.terminal_node
            node = self.root
        return results
