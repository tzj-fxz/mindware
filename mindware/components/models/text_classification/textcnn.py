import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, CategoricalHyperparameter

from mindware.components.models.base_nn import BaseTextClassificationNeuralNetwork
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.datasets.base_dl_dataset import TextDataset, TotalTextDataset


class TextCNNModel(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]

        embedded = self.embedding(text)
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


class TextCNN(BaseTextClassificationNeuralNetwork):

    def __init__(self, embedding_dim=100, n_filters=100, kerlow=3, kerup=6, dropout=0.5,
                 optimizer='Adam', batch_size=64, epoch_num=100, lr_decay=0.1, weight_decay=1e-4,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None, device='cuda',
                 dl_model_path='/tmp/'):
        super(TextCNN, self).__init__(optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                                      sgd_learning_rate, sgd_momentum, nesterov, adam_learning_rate,
                                      beta1, random_state, device, dl_model_path)
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = [ker for ker in range(kerlow, kerup)]
        self.dropout = dropout
        self.random_state = random_state

    def fit(self, dataset: TotalTextDataset, scorer):
        self.vocab_size = len(dataset.train_token_dict)
        self.output_dim = len(np.unique(dataset.train_data.y))
        self.model = TextCNNModel(self.vocab_size, self.embedding_dim,
                                  self.n_filters, self.filter_sizes,
                                  self.output_dim, self.dropout, 0)
        self.model.embedding.weight.data[0] = torch.zeros(self.embedding_dim)
        super().fit(dataset, scorer)
        return self

    def predict(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        if self.model is None:
            raise NotImplementedError()
        return super().predict(dataset, scorer, test=test, model=model)

    def predict_proba(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        print('test')
        if self.model is None:
            raise NotImplementedError()
        return super().predict_proba(dataset, scorer, test=test, model=model)

    def score(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        if self.model is None:
            raise NotImplementedError()
        return super().score(dataset, metric=scorer, test=test, model=model)

    @classmethod
    def get_hyperparameter_search_space(cls, dataset_properties=None, optimizer='smac'):
        cs = super().get_hyperparameter_search_space(dataset_properties, optimizer)
        
        # These HPs are not implemented to be searched yet

        # embedding_dim = CategoricalHyperparameter("embedding_dim", [50, 100, 150, 200], default_value=100)
        # n_filters = CategoricalHyperparameter("n_filters", [50, 100, 150, 200], default_value=100)
        # kerlow = CategoricalHyperparameter("kerlow", [2, 3, 4], default_value=3)
        # kerup = CategoricalHyperparameter("kerup", [6, 7, 8], default_value=6)
        # dropout = UniformFloatHyperparameter("dropout", lower=0, upper=0.6, default_value=0.5, log=False)
        
        embedding_dim = UnParametrizedHyperparameter("embedding_dim", 100)
        n_filters = UnParametrizedHyperparameter("n_filters", 100)
        kerlow = UnParametrizedHyperparameter("kerlow", 3)
        kerup = UnParametrizedHyperparameter("kerup", 6)
        dropout = UnParametrizedHyperparameter("dropout", 0.5)
        cs.add_hyperparameters([embedding_dim, n_filters, kerlow, kerup, dropout])
        return cs
