import numpy as np
import random
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.configspace_utils import check_none

class KNN(Transformer):
    type = 108

    def __init__(self, n_neighbors_knn=20, weights_knn='uniform', ratio_knn=0.2):
        super().__init__('knn')
        self.n_neighbors = n_neighbors_knn
        self.weight = weights_knn
        self.ratio = ratio_knn
        
    # @ease_trans
    def operate(self, input_datanode, target_fields=None):
        # print("KNN!")
        from sklearn.preprocessing import LabelEncoder
        import pandas as pd
        X, y = input_datanode.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_label = LabelEncoder().fit_transform(y)

        if self.model is None:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                weights=self.weight)
            self.model.fit(X, y_label)

            pred_y = self.model.predict_proba(X)
            # print(pred_y)
            num = len(np.unique(y))
            mislabel_y = []
            for i in range(X.shape[0]):
                y_true = np.zeros(num)
                y_true[y_label[i]] = 1 # one hot向量
                pred_label = np.argmax(pred_y[i]) # 众数
                if pred_label != y_label[i]: # 潜在mislabel
                    mislabel_y.append((i, log_loss(y_true, pred_y[i])))
            # print(mislabel_y)
            mislabel_y = sorted(mislabel_y, key=lambda k: k[1], reverse=False)[:int(len(mislabel_y)*self.ratio)]
            changelist = [int(k[0]) for k in mislabel_y]
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
        ratio = UniformFloatHyperparameter('ratio_knn', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        n_neighbors = UniformIntegerHyperparameter(name="n_neighbors_knn", lower=10, upper=100, log=True, default_value=20)
        cs.add_hyperparameter(n_neighbors)
        weight = CategoricalHyperparameter(name="weights_knn", choices=['uniform', 'distance'], default_value='uniform')
        cs.add_hyperparameter(weight)
        return cs
