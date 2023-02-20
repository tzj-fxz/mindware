import numpy as np
import pandas as pd
import copy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from mindware.components.feature_engineering.transformations.base_transformer import *
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition

class LOF(Transformer):
    type = 105

    def __init__(self, n_neighbors=20, algorithm='auto', contamination='auto', novelty=True, ratio_lof=0.2):
        super().__init__('lof')
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.contamination = contamination
        self.novelty = novelty
        self.ratio = ratio_lof
    
    def operate(self, input_datanode, target_fields=None):
        if len(target_fields) == 0:
            return input_datanode
        
        X, y = input_datanode.data
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        # print("LOF: ", X.shape)
        X_input = X[:, target_fields]

        # 先拷贝、归一化 
        X_scaler = copy.deepcopy(X_input)
        X_scaler = MinMaxScaler().fit_transform(X_scaler)
        
        if self.model is None:
            # print("new model here")
            self.model = LocalOutlierFactor(n_neighbors=self.n_neighbors,
                algorithm=self.algorithm,
                contamination=self.contamination,
                novelty=self.novelty)
            self.model.fit(X_scaler)
        
            # 寻找离群值，factor绝对值越大，越可能离群
            pred = -self.model.negative_outlier_factor_
            pred_mean = np.mean(pred)
            # print(mislabel1[0][:5], len(mislabel1[0]))
            sorted_id = sorted(range(len(pred)), key=lambda k:pred[k], reverse=True)
            outlier = []
            for i in sorted_id:
                if pred[i] > max(1.5, pred_mean):
                    outlier.append(i) # 获得可能的mislabel
                else:
                    break
            #print(outlier_y[:30])
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
        ratio = UniformFloatHyperparameter('ratio_lof', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        algorithm = CategoricalHyperparameter("algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
        cs.add_hyperparameter(algorithm)
        contamination = UniformFloatHyperparameter("contamination", 0.0001, 0.5, log=True, default_value=0.1)
        cs.add_hyperparameter(contamination)
        n_neighbors = UniformIntegerHyperparameter("n_neighbors", lower=2, upper=100, log=True, default_value=5)
        cs.add_hyperparameter(n_neighbors)
        return cs
        