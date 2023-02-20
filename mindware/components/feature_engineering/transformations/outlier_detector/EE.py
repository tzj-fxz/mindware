import numpy as np
import pandas as pd
import copy
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler

from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

class EE(Transformer):
    type = 104

    def __init__(self, assume_centered_ell=True, contamination_ell=0.1, ratio_ee=0.2):
        super().__init__('ee')
        self.assume_centered = assume_centered_ell
        self.contamination = contamination_ell
        self.ratio = ratio_ee

    def operate(self, input_datanode, target_fields=None):
        if len(target_fields) == 0:
            return input_datanode

        X, y = input_datanode.data
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        # print("EE: ", X.shape)
        X_input = X[:, target_fields]

        # 先拷贝、归一化
        X_scaler = copy.deepcopy(X_input)
        X_scaler = MinMaxScaler().fit_transform(X_scaler)
        
        if self.model is None:
            # print("new model here")
            self.model = EllipticEnvelope(assume_centered=self.assume_centered,
                contamination=self.contamination)
            self.model.fit(X_scaler)
        
            # decision function判断离群
            pred = self.model.decision_function(X_scaler)
            sorted_id = sorted(range(len(pred)), key=lambda k: pred[k], reverse=False)
            outlier = []
            for i in sorted_id:
                if pred[i] < 0:
                    outlier.append(i)
                else:
                    break
            changelist = sorted_id[:int(len(outlier)*self.ratio)]
            # changelist = [X.iloc[i].name for i in changelist]
            print(changelist[:20])
            new_X, new_y = np.delete(X, changelist, axis=0), np.delete(y, changelist, axis=0)
        else:
            new_X, new_y = X, y
        
        new_feature_types = input_datanode.feature_types.copy()
        output_datanode = DataNode((new_X, new_y), new_feature_types, input_datanode.task_type)
        output_datanode.trans_hist = input_datanode.trans_hist.copy()
        output_datanode.trans_hist.append(self.type)

        return output_datanode
    
    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = ConfigurationSpace()
        ratio = UniformFloatHyperparameter('ratio_ee', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        assume_centered = CategoricalHyperparameter("assume_centered_ell", [True, False], default_value=False)
        cs.add_hyperparameter(assume_centered)
        contamination = UniformFloatHyperparameter("contamination_ell", 0.0001, 0.5, log=True, default_value=0.1)
        cs.add_hyperparameter(contamination)
        return cs
