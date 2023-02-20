import warnings
import copy
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.utils.configspace_utils import check_for_bool, check_none

class IterativeImputer(Transformer):
    type = 103

    def __init__(self, max_iter=5, sample_posterior=False,
        initial_strategy='mean', imputation_order='ascending'):
        super().__init__('iterativeimputernum')
        self.max_iter = max_iter
        self.sample_posterior = sample_posterior
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order

    def operate(self, input_datanode, target_fields=None):
        # print("iterimpute")
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import pandas as pd

        if len(target_fields) == 0:
            return input_datanode

        X, y = input_datanode.data
        # print(target_fields)
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_input = X[:, target_fields]

        if self.model is None:
            self.model = IterativeImputer(max_iter=self.max_iter,
                sample_posterior=self.sample_posterior,
                initial_strategy=self.initial_strategy,
                imputation_order=self.imputation_order)
            self.model.fit(X_input)
        X_new = copy.deepcopy(X)
        X_new[:, target_fields] = self.model.transform(X_input)

        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode((X_new, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        max_iter = UniformIntegerHyperparameter(name="max_iter", lower=1, upper=100, log=True, default_value=5)
        sample_posterior = CategoricalHyperparameter("sample_posterior", choices=[True, False], default_value=False)
        initial_strategy = CategoricalHyperparameter("initial_strategy", ["mean", "median", "most_frequent", "constant"], default_value="mean")
        imputation_order = CategoricalHyperparameter("imputation_order", ["ascending", "descending", "roman", "arabic", "random"], default_value="ascending")
        cs = ConfigurationSpace()
        cs.add_hyperparameters([max_iter,
                                sample_posterior,
                                initial_strategy,
                                imputation_order])
        return cs
