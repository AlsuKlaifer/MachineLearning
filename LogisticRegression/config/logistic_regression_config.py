from easydict import EasyDict

from utils.enums import DataProcessTypes, WeightsInitType, GDStoppingCriteria

cfg = EasyDict()

# data
cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1
cfg.data_preprocess_type = DataProcessTypes.standardization
# cfg.data_preprocess_type = DataProcessTypes.normalization

# training
cfg.weights_init_type = WeightsInitType.normal
cfg.weights_init_kwargs = {'sigma': 1}
# cfg.weights_init_type = WeightsInitType.uniform
# cfg.weights_init_kwargs = {'epsilon': 1}
# cfg.weights_init_type = WeightsInitType.xavier
# cfg.weights_init_kwargs = {'n_in': 1, 'n_out': 1}
# cfg.weights_init_type = WeightsInitType.he
# cfg.weights_init_kwargs = {'n_in': 1}

cfg.gamma = 0.01
cfg.gd_stopping_criteria = GDStoppingCriteria.epoch
cfg.nb_epoch = 100
