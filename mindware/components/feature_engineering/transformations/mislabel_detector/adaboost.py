import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition
from mindware.components.feature_engineering.transformations.base_transformer import *
from mindware.components.utils.configspace_utils import check_none

class Adaboost(Transformer):
    type = 106

    def __init__(self, criterion_ada='gini', n_estimators_ada=100, learning_rate_ada=0.1, \
        max_samples_ada=0.6, ratio_ada=0.3):
        super().__init__('adaboost')
        self.criterion = criterion_ada
        self.n_estimators = n_estimators_ada
        self.learning_rate = learning_rate_ada
        self.max_samples = max_samples_ada
        self.ratio = ratio_ada
    
    # @ease_trans
    def operate(self, input_datanode, target_fields=None):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        X, y = input_datanode.data
        if isinstance(X, pd.DataFrame):
            X = X.values
        y_label = LabelEncoder().fit_transform(y)

        # 没有找到subsample参数，只好随机选取了qaq
        random.seed(1)
        train_index = random.sample(range(len(y)), int(self.max_samples*len(y)))
        if self.model is None:
            self.base_est = DecisionTreeClassifier(criterion=self.criterion, random_state=1)
            self.model = AdaBoostClassifier(base_estimator=self.base_est, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, random_state=1)
            self.model.fit(X[train_index], y_label[train_index])
        
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
        ratio = UniformFloatHyperparameter('ratio_ada', lower=0, upper=0.3, default_value=0.1)
        cs.add_hyperparameter(ratio)
        criterion = CategoricalHyperparameter(name='criterion_ada', choices=['gini', 'entropy'], default_value='gini')
        cs.add_hyperparameter(criterion)
        n_estimators = UniformIntegerHyperparameter(name='n_estimators_ada', lower=2, upper=200, default_value=20)
        cs.add_hyperparameter(n_estimators)
        learning_rate = UniformFloatHyperparameter(name='learning_rate_ada', lower=0.01, upper=0.1, default_value=0.1)
        cs.add_hyperparameter(learning_rate)
        max_samples = UniformFloatHyperparameter(name="max_samples_ada", lower=0.5, upper=0.7, log=True, default_value=0.6)
        cs.add_hyperparameter(max_samples)
        return cs
