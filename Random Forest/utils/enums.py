from enum import IntEnum

DataProcessTypes = IntEnum('DataProcessTypes', ('standardization', 'normalization'))
SetType = IntEnum('SetType', ('train', 'valid', 'test'))
TaskType = IntEnum('TaskTypes', ('classification', 'regression'))
TrainingAlgorithms = IntEnum('TrainingAlgorithms', ('bagging', 'random_node_optimization'))
