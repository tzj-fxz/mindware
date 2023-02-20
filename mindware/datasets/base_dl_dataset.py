import torch
from torch.utils.data import Dataset, DataLoader
import random
import copy


class TextDataset(Dataset):
    def __init__(self, file_location=None, X=None, y=None):
        if X != None or y != None:
            self.X = X
            self.y = y
        else:
            self.X = list()
            self.y = list()
            DataVec = open(file_location, "r", encoding="utf-8").read().split("\n")
            DataVec = list(filter(None, DataVec))
            random.shuffle(DataVec)
            self.DataVec = DataVec
            for item in range(len(DataVec)):
                sent = self.DataVec[item]
                sent = sent.split('\t')
                sent = list(filter(None, sent))
                sentence = sent[0]
                idx = int(sent[-1])
                self.X.append(sentence)
                self.y.append(int(idx))


    def __getitem__(self, item):
        # print(self.X[item], self.y[item])
        if self.y is not None:
            idx = int(self.y[item])
        else:
            idx = 0
        sentence = torch.LongTensor(self.X[item])
        return sentence, idx

    def __len__(self):
        return len(self.X)

    def copy_(self):
        X_ = copy.deepcopy(self.X)
        y_ = copy.deepcopy(self.y)
        return TextDataset(X=X_, y=y_)



class TotalTextDataset():
    def __init__(self,
        train_data: TextDataset,
        val_data: TextDataset,
        test_data: TextDataset,
        train_token_dict: dict
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_token_dict = train_token_dict

    def copy_(self):
        train_data_ = copy.deepcopy(self.train_data)
        val_data_ = copy.deepcopy(self.val_data)
        test_data_ = copy.deepcopy(self.test_data)
        train_token_dict_ = copy.deepcopy(self.train_token_dict)
        return TotalTextDataset(train_data_, val_data_, test_data_, train_token_dict_)
