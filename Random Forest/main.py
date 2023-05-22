import numpy as np

import utils.metrics as metrics
from config.decision_tree_config import cfg as des_tree_cfg
from datasets.digits_dataset import Digits
from models.random_forest import RandomForest
from utils.enums import SetType, TrainingAlgorithms
from utils.visualisation import Visualisation


def experiment_random_forest(digits: Digits):
    # validation
    nb_models = 30
    models = []
    for i in range(nb_models):
        l_1 = np.random.randint(10, 40)
        l_2 = np.random.randint(5, 35)
        m = np.random.randint(5, 20)
        random_forest = RandomForest(nb_trees=m,
                                     max_depth=7,
                                     min_entropy=0.01,
                                     min_number_of_elem=1)
        random_forest.train(training_algorithm=TrainingAlgorithms.random_node_optimization,
                            inputs=digits(SetType.train)['inputs'],
                            targets=digits(SetType.train)['targets'],
                            nb_classes=digits.k,
                            max_nb_dim_to_check=l_1,
                            max_nb_thresholds=l_2)
        predictions_valid = random_forest.get_predictions(digits(SetType.valid)['inputs'])
        confusion_matrix_valid = metrics.confusion_matrix(predictions_valid, digits(SetType.valid)['targets'])
        accuracy_valid = metrics.accuracy(confusion_matrix_valid)
        print(f'Validation: model #{i + 1}, M = {m}, L1 = {l_1}, L2 = {l_2}, accuracy_valid = {accuracy_valid}')
        models.append([random_forest, accuracy_valid, 0])

    # testing best models
    models.sort(key=lambda x: x[1], reverse=True)
    best_models = models[:10]
    for i in range(len(best_models)):
        predictions_test = best_models[i][0].get_predictions(digits(SetType.test)['inputs'])
        confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
        accuracy_test = metrics.accuracy(confusion_matrix_test)
        best_models[i][2] = accuracy_test

    # plot for best models
    Visualisation.visualise_best_models(best_models, plot_title='10 best models')

    # confusion matrix for best model
    best_model = best_models[0][0]
    predictions_test = best_model.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    print()
    print('Confusion matrix for best model on test set:')
    print(confusion_matrix_test)


# BONUS TASK
def experiment_random_forest_bagging(digits: Digits):
    random_forest = RandomForest(nb_trees=10,
                                 max_depth=7,
                                 min_entropy=0.01,
                                 min_number_of_elem=1)
    random_forest.train(training_algorithm=TrainingAlgorithms.bagging,
                        inputs=digits(SetType.train)['inputs'],
                        targets=digits(SetType.train)['targets'],
                        nb_classes=digits.k,
                        subset_size=int(digits(SetType.train)['inputs'].shape[0] * 0.8))
    predictions_test = random_forest.get_predictions(digits(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('BAGGING. Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'BAGGING. Accuracy on test set: {accuracy_test}')


if __name__ == '__main__':
    digits_dataset = Digits(des_tree_cfg)
    experiment_random_forest(digits_dataset)
    print()
    experiment_random_forest_bagging(digits_dataset)
