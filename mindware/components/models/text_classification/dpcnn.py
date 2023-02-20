import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter

from mindware.components.models.base_nn import BaseTextClassificationNeuralNetwork
from mindware.components.utils.constants import DENSE, UNSIGNED_DATA, SPARSE, PREDICTIONS
from mindware.datasets.base_dl_dataset import TextDataset, TotalTextDataset


class DPCNNModel(nn.Module):

    def __init__(self, vocab_size, embed_size, region_embed_size, channel, fc_cla):
        super(DPCNNModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.region = nn.Conv1d(in_channels=embed_size, out_channels=channel, kernel_size=region_embed_size) # like textcnn
        self.padding1 = nn.ConstantPad1d((1, 1), 0)
        self.padding2 = nn.ConstantPad1d((0, 1), 0)
        self.cnn = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(channel, fc_cla)

    def forward(self, x):
        embed = self.embedding(x)
        embed = self.region(embed.transpose(1, 2))
        embed = self.padding1(embed)

        conv = embed + self._block1(self._block1(embed))

        while conv.size(-1) >= 2:
            conv = self._block2(conv)

        out = self.fc(torch.squeeze(conv, dim=-1))
        return out

    def _block1(self, x):
        return self.cnn(self.relu(x))

    def _block2(self, x):
        x = self.padding2(x)
        px = self.maxpooling(x)
        
        x = self.relu(px)
        x = self.cnn(x)

        x = self.relu(x)
        x = self.cnn(x)
        
        x = px + x
        return x


class DPCNN(BaseTextClassificationNeuralNetwork):

    def __init__(self, embed_size=100, region_embed_size=3, channel=250,
                 optimizer='Adam', batch_size=64, epoch_num=100, lr_decay=0.1, weight_decay=1e-4,
                 sgd_learning_rate=None, sgd_momentum=None, nesterov=None,
                 adam_learning_rate=None, beta1=None, random_state=None, device='cuda',
                 dl_model_path='/tmp/'):
        super(DPCNN, self).__init__(optimizer, batch_size, epoch_num, lr_decay, weight_decay,
                                    sgd_learning_rate, sgd_momentum, nesterov, adam_learning_rate,
                                    beta1, random_state, device, dl_model_path)
        self.embed_size = embed_size
        self.region_embed_size = region_embed_size
        self.channel = channel

    def fit(self, dataset: TotalTextDataset, scorer):
        self.vocab_size = len(dataset.train_token_dict)
        self.output_dim = len(np.unique(dataset.train_data.y))
        self.model = DPCNNModel(vocab_size=self.vocab_size, embed_size=self.embed_size, \
            region_embed_size=self.region_embed_size, channel=self.channel, fc_cla=self.output_dim)
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
        
        # embed_size = UniformIntegerHyperparameter("embed_size", lower=100, upper=200, default_value=100)
        # channel = UniformIntegerHyperparameter("channel", lower=200, upper=300, default_value=250)
        # # region_embed_size = UniformIntegerHyperparameter("region_embed_size", lower=2, upper=8, default_value=3)
        
        embed_size = UnParametrizedHyperparameter("embed_size", 100)
        channel = UnParametrizedHyperparameter("channel", 250)
        region_embed_size = UnParametrizedHyperparameter("region_embed_size", 3)

        cs.add_hyperparameters([embed_size, region_embed_size, channel])
        return cs
