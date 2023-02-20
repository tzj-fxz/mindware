import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.configspace_utils import check_none

class RF(Transformer):
    type = 109

    def __init__(self, criterion_rf='gini', n_estimators_rf=100, max_samples_rf=0.6, ratio_rf=0.3):
        super().__init__('rf')
        self.criterion = criterion_rf
        self.n_estimators = n_estimators_rf
        self.max_samples = max_samples_rf
        self.ratio = ratio_rf

    # @ease_trans
    def operate(self, input_datanode, target_fields=None):
        # print("RF!")
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        X, y = input_datanode.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_label = LabelEncoder().fit_transform(y)

        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                criterion=self.criterion,
                max_samples=self.max_samples)
            self.model.fit(X, y_label)
            
            pred_y = self.model.predict_proba(X)
            # print(pred_y)
            num = len(np.unique(y))
            mislabel_y = []
            for i in range(X.shape[0]):
                # print(pred_y[i], y.iloc[i])
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
        ratio = UniformFloatHyperparameter('ratio_rf', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        n_estimators = UniformIntegerHyperparameter(name="n_estimators_rf", lower=2, upper=200, log=True, default_value=20)
        cs.add_hyperparameter(n_estimators)
        criterion = CategoricalHyperparameter(name='criterion_rf', choices=['gini', 'entropy'], default_value='gini')
        cs.add_hyperparameter(criterion)
        max_samples = UniformFloatHyperparameter(name="max_samples_rf", lower=0.5, upper=0.7, log=True, default_value=0.6)
        cs.add_hyperparameter(max_samples)
        return cs
