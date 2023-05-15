import numpy as np

from datasets.wine_quality_dataset import WineQuality
from models.gradient_boosting import GradientBoosting
from config.gradient_boosting_config import cfg as gradient_boosting_cfg
from utils.enums import SetType
from utils.metrics import MSE
from utils.visualisation import Visualisation


def experiment(wine_quality: WineQuality):
    gradient_boosting = GradientBoosting(nb_of_weak_learners=100, weight_of_weak_learners=0.05)
    gradient_boosting.train(inputs=wine_quality(SetType.train)['inputs'],
                            targets=wine_quality(SetType.train)['targets'])
    predictions_test = gradient_boosting.get_predictions(inputs=wine_quality(SetType.test)['inputs'])
    mse = MSE(predictions_test, wine_quality(SetType.test)['targets'])
    print(f'Error value on test set: {mse}')


def experiment_additional(wine_quality: WineQuality):
    # validation
    number_of_models = 30
    models = []
    for i in range(number_of_models):
        m = np.random.randint(low=10, high=51)
        alpha = np.random.uniform(low=0.01, high=0.1)
        gradient_boosting = GradientBoosting(nb_of_weak_learners=m,
                                             weight_of_weak_learners=alpha)
        gradient_boosting.train(inputs=wine_quality(SetType.train)['inputs'],
                                targets=wine_quality(SetType.train)['targets'])
        predictions_valid = gradient_boosting.get_predictions(wine_quality(SetType.valid)['inputs'])
        mse_valid = MSE(predictions_valid, wine_quality(SetType.test)['targets'])
        print(f'Validation: model #{i + 1}, M = {m}, alpha = {alpha}, MSE = {mse_valid}')
        models.append([gradient_boosting, mse_valid, 0])

    # testing best models
    models.sort(key=lambda x: x[1])
    best_models = models[:10]
    for i in range(len(best_models)):
        predictions_test = best_models[i][0].get_predictions(wine_quality(SetType.test)['inputs'])
        mse_test = MSE(predictions_test, wine_quality(SetType.test)['targets'])
        best_models[i][2] = mse_test

    # plot for best models
    Visualisation.visualise_best_models(best_models, plot_title='10 best models')


if __name__ == '__main__':
    data = WineQuality(cfg=gradient_boosting_cfg)
    experiment(data)
    experiment_additional(data)
