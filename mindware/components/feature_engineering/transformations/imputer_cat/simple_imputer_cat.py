import warnings
import copy
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.utils.configspace_utils import check_for_bool, check_none

class SimpleImputerCat(Transformer):
    type = 110

    def __init__(self, strategy_cat='most_frequent'):
        super().__init__('simpleimputercat')
        self.param = strategy_cat

    def operate(self, input_datanode, target_fields=None):
        # print('simpleimpute')
        from sklearn.impute import SimpleImputer
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder

        if len(target_fields) == 0:
            return input_datanode

        X, y = input_datanode.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_input = X[:, target_fields]
        # print(target_fields)
        # print('simple: ', X.shape)
        
        if self.model is None:
            self.model = SimpleImputer(strategy=self.param, copy=False)
            self.model.fit(X_input)
        X_output = copy.deepcopy(X)
        X_output[:, target_fields] = self.model.transform(X_input)
        
        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode((X_output, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        strategy_cat = CategoricalHyperparameter("strategy_cat", ['most_frequent', 'constant'], default_value='most_frequent')
        cs.add_hyperparameter(strategy_cat)
        return cs
