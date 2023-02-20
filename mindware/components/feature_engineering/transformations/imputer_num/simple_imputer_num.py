from operator import indexOf
import warnings
import copy
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.utils.configspace_utils import check_for_bool, check_none

class SimpleImputerNum(Transformer):
    type = 101

    def __init__(self, strategy='mean'):
        super().__init__('simpleimputernum')
        self.param = strategy

    def operate(self, input_datanode, target_fields=None):
        # print('simpleimpute')
        from sklearn.impute import SimpleImputer
        import pandas as pd

        if len(target_fields) == 0:
            return input_datanode

        X, y = input_datanode.data
        # print(target_fields, X.shape)
        # print('simple: ', X.shape)
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_input = X[:, target_fields]
        
        if self.model is None:
            self.model = SimpleImputer(strategy=self.param, copy=False)
            self.model.fit(X_input)
        X_output = copy.deepcopy(X)
        try:
            X_output[:, target_fields] = self.model.transform(X_input)
        except Exception as e:
            print(X_input)
            print(e)
            raise KeyboardInterrupt
        
        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode((X_output, y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        strategy = CategoricalHyperparameter("strategy", ['mean', 'median', 'most_frequent'], default_value='mean')
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs
