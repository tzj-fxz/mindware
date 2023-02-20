import numpy as np
import pandas as pd
import random
from sklearn.ensemble import IsolationForest
from sklearn.metrics import balanced_accuracy_score

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.configspace_utils import check_none

class IF(Transformer):
    type = 107

    def __init__(self, n_estimators_if=100, contamination_if='auto', ratio_if=0.3):
        super().__init__('if')
        self.contamination = contamination_if
        self.n_estimators = n_estimators_if
        self.ratio = ratio_if

    # @ease_trans
    def operate(self, input_datanode, target_fields=None):
        # print("IF!")
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder
        X, y = input_datanode.data
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.model is None:
            self.model = IsolationForest(n_estimators=self.n_estimators,
                contamination=self.contamination)
            self.model.fit(X, y)

            pred_y = self.model.decision_function(X)
            sorted_id = sorted(range(len(pred_y)), key=lambda k:pred_y[k], reverse=False)
            mislabel_y = []
            for i in sorted_id:
                if pred_y[i] < 0:
                    mislabel_y.append(i) # 获得可能的mislabel
                else:
                    break
            changelist = sorted_id[:int(len(mislabel_y)*self.ratio)]
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
        ratio = UniformFloatHyperparameter('ratio_if', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        n_estimators = UniformIntegerHyperparameter(name='n_estimators_if', lower=2, upper=200, default_value=20)
        cs.add_hyperparameter(n_estimators)
        contamination = UniformFloatHyperparameter(name="contamination_if", lower=0.0001, upper=0.5, log=True, default_value=0.1)
        cs.add_hyperparameter(contamination)
        return cs
