from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter
import warnings, os, time
import numpy as np
import pickle as pkl
from sklearn.metrics._scorer import accuracy_scorer, _ThresholdScorer

from mindware.utils.logging_utils import get_logger
from mindware.components.evaluators.base_evaluator import _BaseEvaluator
from mindware.components.evaluators.evaluate_func import nn_validation
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.class_loader import get_combined_candidtates
from mindware.components.models.text_classification import _text_classifiers, _addons
from mindware.components.utils.constants import *
from mindware.datasets.base_dl_dataset import TotalTextDataset

EPOCH_NUM = 100


def get_estimator(config, estimator_id, dl_model_path='/tmp/', resource_ratio=1):
    text_classifier_type = estimator_id
    config_ = config.copy()
    config_['%s:random_state' % text_classifier_type] = 1
    hpo_config = dict()
    for key in config_:
        key_name = key.split(':')[0]
        if key_name == text_classifier_type:
            act_key = key.split(':')[1]
            if act_key == 'epoch_num':
                # partial resource
                hpo_config[act_key] = int(config_[key] * resource_ratio)
            else:
                hpo_config[act_key] = config_[key]

    _candidates = get_combined_candidtates(_text_classifiers, _addons)
    estimator = _candidates[text_classifier_type](**hpo_config, dl_model_path=dl_model_path)
    if hasattr(estimator, 'n_jobs'):
        setattr(estimator, 'n_jobs', 1)
    return text_classifier_type, estimator, hpo_config


def get_hpo_cs(estimator_id, task_type=TEXT_CLS):
    _candidates = get_combined_candidtates(_text_classifiers, _addons)
    if estimator_id in _candidates:
        textclf_class = _candidates[estimator_id]
    else:
        raise ValueError("Algorithm %s not supported!" % estimator_id)
    cs = textclf_class.get_hyperparameter_search_space()
    return cs


def get_cash_cs(include_algorithm=None, task_type=TEXT_CLS):
    _candidates = get_combined_candidtates(_text_classifiers, _addons)
    if include_algorithm is not None:
        _candidates = set(include_algorithm).intersection(set(_candidates.keys()))
        if len(_candidates) == 0:
            raise ValueError("No algorithm included! Please check the spelling of included algorithms!")
    cs = ConfigurationSpace()
    algo = CategoricalHyperparameter("algorithm", list(_candidates))
    cs.add_hyperparameter(algo)

    for estimator_id in _candidates:
        estimator_cs = get_hpo_cs(estimator_id)
        parent_hyperparameter = {"parent": algo, "value": estimator_id}
        cs.add_configuration_space(estimator_id, estimator_cs, parent_hyperparameter=parent_hyperparameter)
    return cs


class TextClassificationEvaluator(_BaseEvaluator):
    def __init__(self, fixed_config=None, scorer=None, dataset=None, task_type=TEXT_CLS, resampling_strategy='cv',
                 resampling_params=None, timestamp=None, output_dir=None, seed=1, if_imbal=False):
        self.resampling_strategy = resampling_strategy
        self.resampling_params = resampling_params

        self.fixed_config = fixed_config
        self.scorer = scorer if scorer is not None else accuracy_scorer
        self.if_imbal = if_imbal
        self.task_type = task_type
        self.dataset = dataset
        self.output_dir = output_dir
        self.seed = seed
        self.onehot_encoder = None
        self.logger = get_logger(self.__module__ + "." + self.__class__.__name__)
        self.continue_training = False

        assert isinstance(dataset, TotalTextDataset)
        self.timestamp = timestamp

    def get_fit_params(self, y, estimator):
        from mindware.components.utils.balancing import get_weights
        _init_params, _fit_params = get_weights(
            y, estimator, None, {}, {}
        )
        return _init_params, _fit_params

    def __call__(self, config, **kwargs):
        start_time = time.time()
        return_dict = dict()
        self.seed = 1
        resource_ratio = kwargs.get('resource_ratio', 1.0)

        # Convert Configuration into dictionary
        if not isinstance(config, dict):
            config = config.get_dictionary().copy()
        else:
            config = config.copy()
        print(config)

        if self.fixed_config is not None:
            config.update(self.fixed_config)
        self.estimator_id = config['algorithm']

        if 'holdout' in self.resampling_strategy:
            # Prepare data node
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

            new_dataset = self.dataset.copy_()

            # No preprocessor
            op_list = {}

            config_dict = config.copy()
            dl_model_path = CombinedTopKModelSaver.get_dl_model_path_by_config(self.output_dir, config, self.timestamp)

            classifier_id, clf, this_config = get_estimator(config_dict, self.estimator_id, dl_model_path)

            score = nn_validation(clf, self.scorer, new_dataset)

            if np.isfinite(score):
                model_para_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)
                if not os.path.exists(model_para_path):
                    with open(model_para_path, 'wb') as f:
                        pkl.dump([op_list, clf, score], f)
                else:
                    with open(model_para_path, 'rb') as f:
                        _, _a, perf = pkl.load(f)
                    if score > perf:
                        with open(model_para_path, 'wb') as f:
                            pkl.dump([op_list, clf, score], f)
                self.logger.info("Model saved to %s and %s" % (model_para_path, dl_model_path))

        elif 'partial' in self.resampling_strategy:
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("ignore")

            new_dataset = self.dataset.copy_()

            # No preprocessor
            op_list = {}

            # Resource -> epoch_num
            config_dict = config.copy()
            dl_model_path = CombinedTopKModelSaver.get_dl_model_path_by_config(self.output_dir, config, self.timestamp)

            classifier_id, clf, this_config = get_estimator(config_dict, self.estimator_id, dl_model_path=dl_model_path,
                                                            resource_ratio=resource_ratio)

            score = nn_validation(clf, self.scorer, new_dataset)

            if np.isfinite(score):
                model_para_path = CombinedTopKModelSaver.get_path_by_config(self.output_dir, config, self.timestamp)
                if not os.path.exists(model_para_path):
                    with open(model_para_path, 'wb') as f:
                        pkl.dump([op_list, clf, score], f)
                else:
                    with open(model_para_path, 'rb') as f:
                        _, _a, perf = pkl.load(f)
                    if score > perf or this_config['epoch_num'] == EPOCH_NUM:
                        with open(model_para_path, 'wb') as f:
                            pkl.dump([op_list, clf, score], f)
                self.logger.info("Model saved to %s and %s" % (model_para_path, dl_model_path))

        else:
            raise ValueError('Invalid resampling strategy: %s!' % self.resampling_strategy)

        try:
            self.logger.info('Evaluation<%s> | Score: %.4f | Time cost: %.2f seconds' %
                             (classifier_id,
                              self.scorer._sign * score,
                              time.time() - start_time))
        except:
            pass

        # Turn it into a minimization problem.
        return_dict['objective'] = -score

        return -score
