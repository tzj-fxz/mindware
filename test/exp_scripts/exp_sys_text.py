"""
    This script is used to compare the strategies/algorithms in the FE-HPO selection
    problem and Bayesian optimization based solution (Auto-scikitlearn)
"""
import os
import sys
import shutil
import time
import pickle
import argparse
import tabulate
import numpy as np
import warnings
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sys.path.append(os.getcwd())
sys.path.append(r'/root/tzj/mindware')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mindware.components.utils.constants import CATEGORICAL, MULTICLASS_CLS, REGRESSION
from mindware.components.utils.text_util import build_dataset, content2Xy, Logger
from mindware.datasets.base_dl_dataset import TotalTextDataset, TextDataset

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
dataset_set = 'abcccc'
parser.add_argument('--datasets', type=str, default=dataset_set)
parser.add_argument('--task_type', type=str, default='text_cls', choices=['text_cls', 'cls', 'rgs'])
parser.add_argument('--mode', type=str, default='alter_hpo')
parser.add_argument('--cv', type=str, choices=['cv', 'holdout', 'partial', 'partial_bohb'], default='holdout')
parser.add_argument('--ens', type=str, default='None')
parser.add_argument('--enable_meta', type=str, default='false', choices=['true', 'false'])
parser.add_argument('--tree_id', type=int, default=0)
parser.add_argument('--time_cost', type=int, default=14400)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--rep_num', type=int, default=1)
# choices=['rb', 'alter_hpo', 'fixed', 'plot', 'all', 'ausk', 'combined']
project_dir = './'
save_folder = project_dir + 'data/exp_sys/'

# prepare for raw data
data_path = "/root/tzj/dataset/工单分类验证数据/全部数据集/"
vocab_path = data_path + 'vocab.txt'
train_path = data_path + 'train.txt'
val_path = data_path + 'dev.txt'
test_path = data_path + 'test.txt'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def evaluate_sys(run_id, task_type, mth, dataset, ens_method, enable_meta,
                 eval_type='holdout', time_limit=1800, seed=1, tree_id=0):
    vocab, train_, val_, test_ = build_dataset(vocab_path, train_path, val_path, test_path, use_word=False)

    train_data_X, train_data_y = content2Xy(train_)
    val_data_X, val_data_y = content2Xy(val_)
    test_data_X, test_data_y = content2Xy(test_)
    train_data = TextDataset(X=train_data_X, y=train_data_y)
    val_data = TextDataset(X=val_data_X, y=val_data_y)
    test_data = TextDataset(X=test_data_X, y=test_data_y)

    tot_dataset = TotalTextDataset(train_data, val_data, test_data, vocab)
    tot_dataset_test = TotalTextDataset(train_data, val_data, test_data, vocab)

    if task_type == 'text_cls':
        from mindware.estimators import TextClassifier
        estimator = TextClassifier(time_limit=time_limit,
                                   dataset_name=dataset,
                                   metric='acc',
                                   include_algorithms=['textcnn'],
                                   output_dir=save_folder,
                                   ensemble_method=ens_method,
                                   ensemble_size=5,
                                   evaluation=eval_type,
                                   per_run_time_limit=int(1e6),
                                   n_jobs=1)

    start_time = time.time()
    estimator.fit(tot_dataset, opt_strategy=mth, dataset_id=dataset, tree_id=tree_id)

    info_dict = estimator.save_info()
    print("save successfully")

    # Test for storage
    new_estimator = TextClassifier(time_limit=time_limit,
                                   dataset_name=dataset,
                                   metric='acc',
                                   include_algorithms=['textcnn'],
                                   output_dir=save_folder,
                                   ensemble_method=ens_method,
                                   ensemble_size=5,
                                   evaluation=eval_type,
                                   per_run_time_limit=int(1e6),
                                   n_jobs=1)

    content_dict = new_estimator.load_info(estimator.output_dir)
    print("content_dict: ", content_dict)

    best_model = estimator._ml_engine.solver.incumbent
    pred_val = new_estimator.predict(tot_dataset, mode='val')
    pred_test = new_estimator.predict(tot_dataset_test, mode='test')

    y_label_val = np.array(val_data_y)
    y_label = np.array(test_data_y)
    validation_score = accuracy_score(y_label_val, pred_val)
    test_score = accuracy_score(y_label, pred_test)

    print('Run ID         : %d' % run_id)
    print('Dataset        : %s' % dataset)
    print('Val/Test score : %f - %f' % (validation_score, test_score))
    print('Val score (best single model) : %f' % (info_dict['best_config_perf']))

    save_path = save_folder + '%s_%s_%s_%s_%d_%d_%d_%d_new_algo_space.pkl' % (
        task_type, mth, dataset, enable_meta, time_limit, (ens_method is None), tree_id, run_id + 7)
    with open(save_path, 'wb') as f:
        pickle.dump([dataset, validation_score, test_score, start_time, best_model], f)

    # Delete output dir
    # shutil.rmtree(os.path.join(estimator.get_output_dir()))


if __name__ == "__main__":
    args = parser.parse_args()
    dataset_str = args.datasets
    time_cost = args.time_cost
    mode = args.mode
    task_type = args.task_type
    tree_id = args.tree_id
    ens_method = args.ens
    if ens_method == 'None':
        ens_method = None
    cv = args.cv
    np.random.seed(1)
    rep = args.rep_num
    start_id = args.start_id
    enable_meta = args.enable_meta
    seeds = np.random.randint(low=1, high=10000, size=start_id + rep)
    dataset_list = dataset_str.split(',')

    if mode == 'all':
        methods = ['rb', 'fixed', 'alter_hpo']
    else:
        methods = [mode]

    for dataset in dataset_list:
        for method in methods:
            for _id in range(start_id, start_id + rep):
                seed = seeds[_id]
                print('Running %s with %d-th seed' % (dataset, _id + 1))
                if method in ['rb', 'fixed', 'alter_hpo', 'combined', 'rb_hpo']:
                    evaluate_sys(_id, task_type, method, dataset, ens_method, enable_meta,
                                 eval_type=cv, time_limit=time_cost, seed=seed, tree_id=tree_id)
                else:
                    raise ValueError('Invalid mode: %s!' % method)
