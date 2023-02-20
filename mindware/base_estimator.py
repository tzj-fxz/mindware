import os
import sys
import traceback
import time
import numpy as np
import pandas as pd
import pickle as pkl

from mindware.automl import AutoML
from mindware.autodl import AutoDL
from mindware.components.feature_engineering.transformation_graph import DataNode
from mindware.datasets.base_dl_dataset import TotalTextDataset, TextDataset
from mindware.components.utils.topk_saver import CombinedTopKModelSaver
from mindware.components.utils.constants import TEXT_CLS, IMG_CLS


class BaseEstimator(object):
    def __init__(
            self,
            dataset_name='default_name',
            time_limit=300,
            amount_of_resource=None,
            metric='acc',
            include_algorithms=None,
            include_preprocessors=None,
            enable_meta_algorithm_selection=True,
            enable_fe=True,
            optimizer='smac',
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            per_run_time_limit=150,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            resampling_params=None,
            output_dir="/tmp/",
            delete_output_dir_after_fit=False):
        self.dataset_name = dataset_name
        self.metric = metric
        self.task_type = None
        self.time_limit = time_limit
        self.amount_of_resource = amount_of_resource
        self.include_algorithms = include_algorithms
        self.include_preprocessors = include_preprocessors
        self.enable_meta_algorithm_selection = enable_meta_algorithm_selection
        self.enable_fe = enable_fe
        self.optimizer = optimizer
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.per_run_time_limit = per_run_time_limit
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.resampling_params = resampling_params
        self._ml_engine = None
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir_postfix = "mindware-%s-%s" % (self.dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.output_dir = os.path.join(output_dir, output_dir_postfix)
        # Create output directory.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.delete_output_dir = delete_output_dir_after_fit

    def get_output_dir(self):
        return self.output_dir

    def build_engine(self):
        """Build AutoML controller"""
        engine = self.get_automl()(
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            amount_of_resource=self.amount_of_resource,
            include_algorithms=self.include_algorithms,
            include_preprocessors=self.include_preprocessors,
            enable_meta_algorithm_selection=self.enable_meta_algorithm_selection,
            enable_fe=self.enable_fe,
            optimizer=self.optimizer,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            per_run_time_limit=self.per_run_time_limit,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            resampling_params=self.resampling_params,
            output_dir=self.output_dir
        )
        return engine

    def initialize(self, data: DataNode, **kwargs):
        assert data is not None and isinstance(data, DataNode)
        self._ml_engine = self.build_engine()
        self._ml_engine.initialize(data, **kwargs)

    def fit(self, data: DataNode, **kwargs):
        assert data is not None and isinstance(data, DataNode)
        self._ml_engine.fit(data, **kwargs)
        # if self.delete_output_dir:
        #     shutil.rmtree(self.output_dir)
        return self

    def predict(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict(X)

    def score(self, data: DataNode):
        return self._ml_engine.score(data)

    def refit(self):
        return self._ml_engine.refit()

    def predict_proba(self, X: DataNode, batch_size=None, n_jobs=1):
        return self._ml_engine.predict_proba(X)

    def get_automl(self):
        return AutoML

    def get_evaluator(self):
        if self._ml_engine is None:
            raise AttributeError("Please initialize the estimator first!")
        return self._ml_engine.solver.evaluator

    def get_config_space(self):
        if self._ml_engine is None:
            raise AttributeError("Please initialize the estimator first!")
        return self._ml_engine.solver.joint_cs

    def show_info(self):
        raise NotImplementedError()

    @property
    def best_hpo_config(self):
        return self._ml_engine.solver.best_hpo_config

    @property
    def best_algo_id(self):
        return self._ml_engine.solver.optimal_algo_id

    @property
    def nbest_algo_id(self):
        return self._ml_engine.solver.nbest_algo_ids

    @property
    def best_perf(self):
        return self._ml_engine.solver.incumbent_perf

    @property
    def best_node(self):
        return self._ml_engine.solver.best_data_node

    @property
    def best_fe_config(self):
        return self._ml_engine.solver.best_data_node.config

    def data_transform(self, data: DataNode):
        return self._ml_engine.solver.fe_optimizer.apply(data, self._ml_engine.solver.best_data_node)

    def feature_corelation(self, data: DataNode):
        X0, y0 = data.data
        X, y = self.data_transformer(data).data
        i = X0.shape[1]
        j = X.shape[1]
        corre_mat = np.zeros([i, j])
        for it in range(i):
            for jt in range(j):
                corre_mat[it, jt] = np.corrcoef(X0[:, it], X[:, jt])[0, 1]
        df = pd.DataFrame(corre_mat)
        df.columns = ['origin_feature' + str(it) for it in range(i)]
        df.index = ['transformed_feature' + str(jt) for jt in range(j)]
        return df

    def feature_origin(self):
        conf = self._ml_engine.solver.best_data_node.config
        pro_table = []
        for process in ['preprocessor1', 'preprocessor2', 'balancer', 'rescaler', 'generator', 'selector']:
            if (conf[process] == 'empty'):
                pro_hash = {'Processor': process, 'Algorithm': None, 'File_path': None, 'Arguments': None}
                pro_table.append(pro_hash)
                continue

            pro_hash = {'Processor': process, 'Algorithm': conf[process]}
            argstr = ''
            for key in conf:
                if (key.find(conf[process]) != -1):
                    arg = key.replace(conf[process] + ':', '')
                    argstr += (arg + '=' + str(conf[key]) + '  ')
            pro_hash['Arguments'] = argstr
            pathstr = './mindware/components/feature_engineering/transformations/'
            if (process == 'preprocessor1'):
                pro_hash['File_path'] = pathstr + 'continous_discretizer.py'
                pro_table.append(pro_hash)
                continue

            if (process == 'preprocessor2'):
                pro_hash['File_path'] = pathstr + 'discrete_categorizer.py'
                pro_table.append(pro_hash)
                continue

            if (process == 'balancer'):
                pro_hash['File_path'] = pathstr + 'preprocessor/' + conf[process] + '.py'
                pro_table.append(pro_hash)
                continue

            pro_hash['File_path'] = pathstr + process + '/' + conf[process] + '.py'
            pro_table.append(pro_hash)

        df = pd.DataFrame(pro_table)[['Processor', 'Algorithm', 'File_path', 'Arguments']]
        df.index = ['step' + str(i) for i in range(1, 7)]
        return df

    def get_val_stats(self):
        return self._ml_engine.get_val_stats()

    def get_ens_model_info(self):
        return self._ml_engine.get_ens_model_info()

    def summary(self):
        return self._ml_engine.summary()


class BaseDLEstimator(object):
    def __init__(
            self,
            dataset_name='default_dataset_name',
            time_limit=1800,
            metric='acc',
            include_algorithms=None,
            ensemble_method='ensemble_selection',
            ensemble_size=50,
            max_epoch=150,
            skip_profile=False,
            config_file_path=None,
            random_state=1,
            n_jobs=1,
            evaluation='holdout',
            output_dir="/tmp/",
            per_run_time_limit=3600,
    ):
        self.dataset_name = dataset_name
        self.metric = metric
        self.task_type = TEXT_CLS
        self.time_limit = time_limit
        self.include_algorithms = include_algorithms
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.max_epoch = max_epoch
        self.skip_profile = skip_profile
        self.config_file_path = config_file_path
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluation = evaluation
        self.output_dir = output_dir
        self._ml_engine = None
        self.timestamp = None
        self.topk_pkl = None
        self.model_idx = None
        self.model_weight = None
        self.best_config = None
        self.per_run_time_limit = per_run_time_limit
        # Create output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dir_postfix = "mindware-%s-%s" % (self.dataset_name, time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.output_dir = os.path.join(output_dir, output_dir_postfix)
        # Create output directory.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def build_engine(self):
        """Build AutoDL controller"""
        engine = self.get_autodl()(
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            metric=self.metric,
            time_limit=self.time_limit,
            include_algorithms=self.include_algorithms,
            ensemble_method=self.ensemble_method,
            ensemble_size=self.ensemble_size,
            max_epoch=self.max_epoch,
            config_file_path=self.config_file_path,
            skip_profile=self.skip_profile,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            evaluation=self.evaluation,
            per_run_time_limit=self.per_run_time_limit,
            output_dir=self.output_dir,
            timestamp=self.timestamp,
            topk_pkl=self.topk_pkl,
            best_config=self.best_config,
            model_idx=self.model_idx,
            model_weight=self.model_weight
        )
        return engine

    def fit(self, data: TotalTextDataset, **kwargs):
        try:
            assert data is not None
            self._ml_engine = self.build_engine()
            self._ml_engine.fit(data, **kwargs)
        except Exception as e:
            print(e)
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
        return self

    def predict(self, dataset: TotalTextDataset, mode='test'):
        assert dataset is not None
        if self.topk_pkl is not None:
            self._ml_engine = self.build_engine()
        if mode == 'val':
            dataset.test_data = dataset.val_data
        return self._ml_engine.predict(dataset)

    def score(self, data, mode='test'):
        assert data is not None
        if self.topk_pkl is not None:
            self._ml_engine = self.build_engine()
        return self._ml_engine.score(data, mode=mode)

    def refit(self, data):
        return self._ml_engine.refit(data)

    def predict_proba(self, dataset: TotalTextDataset, mode='test', batch_size=1, n_jobs=1):
        assert dataset is not None
        if self.topk_pkl is not None:
            self._ml_engine = self.build_engine()
        return self._ml_engine.predict_proba(dataset)

    def get_runtime_history(self):
        return self._ml_engine._get_runtime_info()

    def get_autodl(self):
        return AutoDL

    def save_info(self):
        print('begin save info')
        info_path = self.output_dir + 'estimator_info.pkl'
        info_dict = dict()
        info_dict['timestamp'] = self._ml_engine.timestamp
        info_dict['output_dir'] = self.output_dir
        info_dict['topk_pkl_path'] = self.output_dir + '/' + str(info_dict['timestamp']) + '_topk_config.pkl'
        if self.ensemble_method == 'ensemble_selection':
            info_dict['best_config'] = self._ml_engine.solver.incumbent
            info_dict['best_config_perf'] = self._ml_engine.solver.incumbent_perf
            info_dict['best_config_path'] = CombinedTopKModelSaver.get_path_by_config(self.output_dir,
                                                                                      self._ml_engine.solver.incumbent,
                                                                                      self._ml_engine.timestamp)
            info_dict['ensemble_model_list_in_topk'] = self._ml_engine.solver.es.model.model_idx
            info_dict['ensemble_model_weight_list_in_topk'] = self._ml_engine.solver.es.model.weights_
            info_dict['ensemble_config_list_with_weight'] = list()
            cur_idx = 0
            for algo_id in self._ml_engine.solver.es.model.stats.keys():
                model_to_save = self._ml_engine.solver.es.model.stats[algo_id]
                for idx, (config_dict, _, model_path) in enumerate(model_to_save):
                    with open(model_path, 'rb') as f:
                        op_list, estimator, _ = pkl.load(f)
                        if cur_idx in self._ml_engine.solver.es.model.model_idx:
                            info_dict['ensemble_config_list_with_weight'].append(
                                (config_dict, self._ml_engine.solver.es.model.weights_[cur_idx]))
                    cur_idx += 1
        else:
            info_dict['best_config'] = self._ml_engine.solver.incumbent
            info_dict['best_config_perf'] = self._ml_engine.solver.incumbent_perf
            info_dict['best_config_path'] = CombinedTopKModelSaver.get_path_by_config(self.output_dir,
                                                                                      self._ml_engine.solver.incumbent,
                                                                                      self._ml_engine.timestamp)
            info_dict['ensemble_model_list_in_topk'] = None
            info_dict['ensemble_model_weight_list_in_topk'] = None
        with open(info_path, 'wb') as f:
            pkl.dump(info_dict, f)
        return info_dict

    def load_info(self, output_dir):
        info_path = output_dir + 'estimator_info.pkl'
        with open(info_path, 'rb') as f:
            content_dict = dict(pkl.load(file=f))
            self.timestamp = content_dict['timestamp']
            self.output_dir = content_dict['output_dir']
            self.topk_pkl = content_dict['topk_pkl_path']
            self.best_config = content_dict['best_config']
            if self.ensemble_method is not None:
                self.model_idx = content_dict['ensemble_model_list_in_topk']
                self.model_weight = content_dict['ensemble_model_weight_list_in_topk']
        return content_dict
