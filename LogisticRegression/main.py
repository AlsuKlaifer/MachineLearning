import utils.metrics as metrics
from config.logistic_regression_config import cfg as log_reg_cfg
from datasets.digits_dataset import Digits
from models.logistic_regression_model import LogReg
from utils.enums import SetType
from utils.visualisation import Visualisation


def experiment():
    digits_dataset = Digits(log_reg_cfg)
    log_reg_model = LogReg(log_reg_cfg, digits_dataset.k, digits_dataset.d)
    log_reg_model.train(digits_dataset(SetType.train)['inputs'], digits_dataset(SetType.train)['targets'],
                        digits_dataset(SetType.valid)['inputs'], digits_dataset(SetType.valid)['targets'])
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['target_function_value_train'],
                                    plot_title='Target function values on training set',
                                    y_title='Target function value')
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['accuracy_train'],
                                    plot_title='Accuracy values on training set',
                                    y_title='Accuracy')
    Visualisation.visualise_metrics(epochs=log_reg_model.data_for_plots['epochs'],
                                    metrics=log_reg_model.data_for_plots['accuracy_valid'],
                                    plot_title='Accuracy values on validation set',
                                    y_title='Accuracy')
    predictions_test = log_reg_model(digits_dataset(SetType.test)['inputs'])
    confusion_matrix_test = metrics.confusion_matrix(predictions_test, digits_dataset(SetType.test)['targets'])
    accuracy_test = metrics.accuracy(confusion_matrix_test)
    print('Confusion matrix on test set:')
    print(confusion_matrix_test)
    print(f'Accuracy on test set: {accuracy_test}')


if __name__ == '__main__':
    experiment()

