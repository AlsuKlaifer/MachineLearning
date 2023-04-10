import utils.metrics as metrics
from config.decision_tree_config import cfg as des_tree_cfg
from datasets.digits_dataset import Digits
from datasets.wine_quality_dataset import WineQuality
from models.decision_tree import DT
from utils.enums import SetType, TaskType


def experiment_classification():
    digits = Digits(des_tree_cfg)
    decision_tree = DT(task_type=TaskType.classification,
                       max_depth=7,
                       min_entropy=0.01,
                       min_number_of_elem=1)
    decision_tree.train(digits(SetType.train)['inputs'], digits(SetType.train)['targets'])

    predictions_valid = decision_tree.get_predictions(digits(SetType.valid)['inputs'])
    confusion_matrix_valid = metrics.confusion_matrix(predictions_valid, digits(SetType.valid)['targets'])
    accuracy_valid = metrics.accuracy(confusion_matrix_valid)
    print('Confusion matrix on valid set:')
    print(confusion_matrix_valid)
    print(f'Accuracy on valid set: {accuracy_valid}')

    print()

    predictions_test = decision_tree.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'Accuracy on test set: {accuracy_test}')


def experiment_regression():
    wine_quality = WineQuality(des_tree_cfg)
    decision_tree = DT(task_type=TaskType.regression,
                       max_depth=10,
                       min_entropy=0.01,
                       min_number_of_elem=1)
    decision_tree.train(wine_quality(SetType.train)['inputs'], wine_quality(SetType.train)['targets'])

    predictions_valid = decision_tree.get_predictions(wine_quality(SetType.valid)['inputs'])
    mean_squared_error_valid = metrics.MSE(predictions_valid, wine_quality(SetType.valid)['targets'])
    print(f'Error value on valid set: {mean_squared_error_valid}')

    predictions_test = decision_tree.get_predictions(wine_quality(SetType.test)['inputs'])
    # targets_test = wine_quality(SetType.test)['targets']
    # for i in range(len(targets_test)):
    #     print(f'{targets_test[i]} {predictions_test[i]}')
    mean_squared_error_test = metrics.MSE(predictions_test, wine_quality(SetType.test)['targets'])
    print(f'Error value on test set: {mean_squared_error_test}')


if __name__ == '__main__':
    experiment_classification()
    print()
    experiment_regression()
