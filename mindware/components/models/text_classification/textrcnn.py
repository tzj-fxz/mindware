import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter, \
    CategoricalHyperparameter

from mindware.components.models.base_nn import BaseTextClassificationNeuralNetwork
from mindware.components.utils.constants import DENSE, SPARSE, UNSIGNED_DATA, PREDICTIONS
from mindware.datasets.base_dl_dataset import TextDataset, TotalTextDataset


class TextRCNNModel(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, layer, bidirection, drop_rate, output_dim):
        super(TextRCNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.hidden_size = hidden_size
        self.layer = layer
        self.bidirection = bidirection
        self.drop_rate = drop_rate
        if bidirection:
            self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=True, num_layers=layer, batch_first=True)
            self.fc = nn.Linear(2*hidden_size+embed_size, output_dim)
        else:
            self.rnn = nn.LSTM(embed_size, hidden_size, bidirectional=False, num_layers=layer, batch_first=True)
            self.fc = nn.Linear(hidden_size+embed_size, output_dim)

    def forward(self, x_in):
        emb = self.embedding(x_in)
        out, _ = self.rnn(emb)
        out = torch.cat([emb, out], dim=-1)
        out = F.relu(out).transpose(1, 2)
        out = F.max_pool1d(out, out.size(-1)).squeeze(-1)
        out = self.fc(out)
        return out


class TextRCNN(BaseTextClassificationNeuralNetwork):
    
    def __init__(self, embed_size=100, hidden_size=10, layer=2, bidirection=False, drop_rate=0.5,
                 optimizer='Adam', batch_size=64, epoch_num=100, lr_decay=0.1, weight_decay=1e-4,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None, device='cuda',
                 dl_model_path='/tmp/'):
        super(TextRCNN, self).__init__(optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                                sgd_learning_rate, sgd_momentum, nesterov, adam_learning_rate,
                                beta1, random_state, device, dl_model_path)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.layer = layer
        self.bidirection = bidirection
        self.drop_rate = drop_rate

    def fit(self, dataset: TotalTextDataset, scorer):
        self.vocab_size = len(dataset.train_token_dict)
        self.output_dim = len(np.unique(dataset.train_data.y))
        self.model = TextRCNNModel(vocab_size=self.vocab_size, embed_size=self.embed_size,
                                   hidden_size=self.hidden_size, layer=self.layer,
                                   bidirection=self.bidirection, drop_rate=self.drop_rate,
                                   output_dim=self.output_dim)
        super().fit(dataset, scorer)
        return self
    
    def predict(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        if self.model is None:
            raise NotImplementedError()
        return super().predict(dataset, scorer, test=test, model=model)

    def predict_proba(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        if self.model is None:
            raise NotImplementedError()
        return super().predict_proba(dataset, scorer, test=test, model=model)

    def score(self, dataset: TotalTextDataset, scorer, test=True, model=None):
        if self.model is None:
            raise NotImplementedError()
        return super().score(dataset, metric=scorer, test=test, model=model)

    @classmethod
    def get_hyperparameter_search_space(dataset_properties=None, optimizer='smac'):
        cs = super().get_hyperparameter_search_space(dataset_properties, optimizer)
        
        # These HPs are not implemented to be searched yet
        
        # embed_size = CategoricalHyperparameter("embed_size", [50, 100, 150, 200], default_value=100)
        # hidden_size = CategoricalHyperparameter("hidden_size", [32, 64, 128], default_value=64)
        # layer = CategoricalHyperparameter("layer", [1, 2], default_value=2)
        # drop_rate = UniformFloatHyperparameter("drop_rate", lower=0, upper=0.6, default_value=0.5, log=False)

        embed_size = UnParametrizedHyperparameter("embed_size", 100)
        hidden_size = UnParametrizedHyperparameter("hidden_size", 64)
        layer = UnParametrizedHyperparameter("layer", 2)
        drop_rate = UnParametrizedHyperparameter("drop_rate", 0.5)

        cs.add_hyperparameters([embed_size, hidden_size, layer, drop_rate])
        return cs
