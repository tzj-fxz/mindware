import os
import time
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.metrics.metric import get_metric
from mindware.components.utils.constants import IMG_CLS, TEXT_CLS, SUCCESS
from mindware.components.ensemble import ensemble_list
from mindware.components.models.text_classification import _text_classifiers
from mindware.blocks.block_utils import get_node_type, get_execution_tree
from mindware.datasets.base_dl_dataset import TotalTextDataset
from ConfigSpace.configuration_space import ConfigurationSpace

text_classification_algorithms = _text_classifiers.keys()


class AutoDL(object):
    def __init__(self, time_limit=300,
                 dataset_name='default_name',
                 amount_of_resource=None,
                 task_type=None,
                 metric='bal_acc',
                 include_algorithms=None,
                 include_preprocessors=None,
                 optimizer='smac',
                 ensemble_method='ensemble_selection',
                 per_run_time_limit=1200,
                 ensemble_size=50,
                 evaluation='holdout',
                 resampling_params=None,
                 output_dir="logs",
                 logging_config=None,
                 random_state=1,
                 n_jobs=1,
                 max_epoch=100,
                 config_file_path=None,
                 skip_profile=None,
                 timestamp=None,
                 topk_pkl=None,
                 model_idx=None,
                 model_weight=None,
                 best_config=None
                 ):
        self.metric_id = metric
        self.metric = get_metric(self.metric_id)

        self.dataset_name = dataset_name
        self.time_limit = time_limit
        self.seed = random_state
        self.per_run_time_limit = per_run_time_limit
        self.output_dir = output_dir
        self.logging_config = logging_config
        self.logger = self._get_logger(self.dataset_name)

        self.evaluation_type = evaluation
        self.resampling_params = resampling_params
        self.include_preprocessors = include_preprocessors

        self.amount_of_resource = int(1e8) if amount_of_resource is None else amount_of_resource
        self.optimizer = optimizer
        self.ensemble_method = ensemble_method
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.solver = None

        self.global_start_time = time.time()
        self.eval_time = None
        self.total_time = None
        self.max_epoch = max_epoch
        self.config_file_path = config_file_path
        self.skip_profile = skip_profile

        self.timestamp = timestamp
        self.topk_pkl = topk_pkl
        self.model_idx = model_idx
        self.model_weight = model_weight
        self.best_config = best_config

        # Disable meta learning
        if self.include_preprocessors is not None:
            self.enable_meta_algorithm_selection = False

        if include_algorithms is not None:
            self.include_algorithms = include_algorithms
        else:
            if task_type not in [IMG_CLS, TEXT_CLS]:
                raise ValueError('Please use AutoML module, instead of AutoDL.')
            self.include_algorithms = list(text_classification_algorithms)
        if ensemble_method is not None and ensemble_method not in ensemble_list:
            raise ValueError("%s is not supported for ensemble!" % ensemble_method)

    def _get_logger(self, name):
        logger_name = 'MindWare-%s(%d)' % (name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)),
                     self.logging_config)
        return get_logger(logger_name)

    def initialize(self, train_data: TotalTextDataset, **kwargs):
        # Check whether this dataset is balanced or not.
        # if self.task_type in CLS_TASKS and is_imbalanced_dataset(train_data):
        #     self.logger.info('Input dataset is imbalanced!')
        #     train_data = DataBalancer().operate(train_data)

        dataset_id = kwargs.get('dataset_id', None)

        # Not support meta_learning

        if self.task_type == TEXT_CLS:
            from mindware.components.evaluators.text_cls_evaluator import get_cash_cs
            self.if_imbal = False
            self.fe_config_space = ConfigurationSpace()
            self.cash_config_space = get_cash_cs(self.include_algorithms, self.task_type)

        # TODO: Define execution trees flexibly
        tree_id = kwargs.get("tree_id", 0)
        tree = get_execution_tree(tree_id)
        solver_type = get_node_type(tree, 0)
        if self.timestamp is None:
            self.timestamp = time.time()
        # print('automl task type: ',self.task_type)
        self.solver = solver_type(tree, 0, self.task_type, self.timestamp,
                                  self.fe_config_space, self.cash_config_space, train_data,
                                  per_run_time_limit=self.per_run_time_limit,
                                  dataset_name=self.dataset_name,
                                  optimizer=self.optimizer,
                                  ensemble_method=self.ensemble_method,
                                  ensemble_size=self.ensemble_size,
                                  metric=self.metric,
                                  seed=self.seed,
                                  time_limit=self.time_limit,
                                  trial_num=self.amount_of_resource,
                                  eval_type=self.evaluation_type,
                                  resampling_params=self.resampling_params,
                                  output_dir=self.output_dir,
                                  n_jobs=self.n_jobs,
                                  topk_pkl=self.topk_pkl,
                                  model_idx=self.model_idx,
                                  model_weight=self.model_weight,
                                  incumbent=self.best_config
                                  )

    def fit(self, train_data, **kwargs):
        """
        This function includes this following two procedures.
            1. tune each algorithm's hyperparameters.
            2. engineer each algorithm's features automatically.
        :param train_data:
        :return:
        """
        if self.solver is None:
            self.initialize(train_data, **kwargs)

        for i in range(self.amount_of_resource):
            if not (self.solver.early_stop_flag or self.solver.timeout_flag):
                self.solver.iterate()
        self.eval_time = time.time() - self.timestamp

        if self.ensemble_method is not None and self.evaluation_type in ['holdout', 'partial']:
            self.solver.fit_ensemble()
        self.total_time = time.time() - self.global_start_time

    def refit(self):
        self.solver.refit()

    def predict_proba(self, test_data: TotalTextDataset, **kwargs):
        if self.solver is None:
            self.initialize(test_data, **kwargs)
        return self.solver.predict_proba(test_data)

    def predict(self, test_data: TotalTextDataset, **kwargs):
        if self.solver is None:
            self.initialize(test_data, **kwargs)
        return self.solver.predict(test_data)

    def score(self, test_data: TotalTextDataset, metric_func=None):
        if metric_func is None:
            metric_func = self.metric
        return metric_func(self, test_data, test_data.data[1])

    def get_ens_model_info(self):
        if self.ensemble_method is not None:
            return self.solver.es.get_ens_model_info()
        else:
            return None

    def get_val_stats(self):
        return self.solver.get_stats()

    def summary(self):
        from terminaltables import AsciiTable
        incumbent = self.solver.incumbent
        if not incumbent:
            return 'No incumbents in history. Please run fit() first.'
        configs_table = []
        nil = "-"
        parameters = list(incumbent.copy().keys())
        for para in parameters:
            row = []
            row.append(para)
            val = incumbent.get(para, None)
            if val is None:
                val = nil
            if isinstance(val, float):
                val = "%.6f" % val
            elif not isinstance(val, str):
                val = str(val)
            row.append(val)
            configs_table.append(row)
        configs_title = ["Parameters", "Optimal Value"]

        total_eval_dict = self.solver.eval_dict.copy()
        num_configs = len(self.solver.eval_dict)
        failed_configs = 0
        for key in total_eval_dict:
            perf, _, state = total_eval_dict[key]
            if state != SUCCESS:
                failed_configs += 1

        table_data = ([configs_title] +
                      configs_table +
                      [["Optimal Validation Performance", self.solver.incumbent_perf]] +
                      [['Number of Configurations', num_configs]] +
                      [['Number of Failed Configurations', failed_configs]] +
                      [['Search Runtime', '%.3f sec' % self.eval_time]] +  # TODO: Precise search time.
                      [['Total Runtime', '%.3f sec' % self.total_time]] +
                      [['Average Evaluation Time', 0]] +  # TODO: Wait for OpenBOX
                      [['Maximum Valid Evaluation Time', 0]] +
                      [['Minimum Evaluation Time', 0]]
                      )

        M = 8
        raw_table = AsciiTable(
            table_data
            # title="Result of Optimization"
        ).table
        lines = raw_table.splitlines()
        title_line = lines[1]
        st = title_line.index("|", 1)
        col = "Optimal Value"
        L = len(title_line)
        lines[0] = "+" + "-" * (L - 2) + "+"
        new_title_line = title_line[:st + 1] + (" " + col + " " * (L - st - 3 - len(col))) + "|"
        lines[1] = new_title_line
        bar = "\n" + lines.pop() + "\n"
        finals = lines[-M:]
        prevs = lines[:-M]
        render_table = "\n".join(prevs) + bar + bar.join(finals) + bar
        return render_table
