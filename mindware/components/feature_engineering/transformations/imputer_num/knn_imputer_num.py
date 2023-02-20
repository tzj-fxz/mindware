import warnings
import copy
from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.utils.configspace_utils import check_for_bool, check_none

class KNNImputer(Transformer):
    type = 102

    def __init__(self, n_neighbors=5, weights='uniform'):
        super().__init__('knnimputernum')
        self.n_neighbors = n_neighbors
        self.weights = weights

    def operate(self, input_datanode, target_fields=None):
        # print("knnimpute")
        from sklearn.impute import KNNImputer
        import pandas as pd

        if len(target_fields) == 0:
            return input_datanode

        X, y = input_datanode.data
        # print(target_fields)
        if isinstance(X, pd.DataFrame):
            X = X.values
        # print("knn: ", X.shape)
        X_input = X[:, target_fields]
            
        if self.model is None:
            self.model = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
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
        n_neighbors = UniformIntegerHyperparameter(name="n_neighbors", lower=2, upper=100, log=True, default_value=5)
        weights = CategoricalHyperparameter(name="weights", choices=["uniform", "distance"], default_value="uniform")
        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_neighbors, weights])
        return cs
