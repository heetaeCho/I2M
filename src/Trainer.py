import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Model import Model

class Trainer:
    def __init__(self, modelType, numCell, batchSize, dropout, maxLen):
        self.modelType = modelType
        self.numCell = numCell
        self.batchSize = batchSize
        self.dropout = dropout
        self.maxLen = maxLen

    def fit(self, project, epoch, data, trainY, numClass):
        dataset = TensorDataset(data, trainY)
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=True)
        model = Model(self.modelType, numClass, self.numCell, self.dropout, self.maxLen)

        loss = 0
        for e in range(epoch+1):
            for idx, samples in enumerate(loader):
                x, y = samples
                x = torch.flip(x, dims=[1]).cuda()
                y = y.view(-1).cuda()
                loss += model.fit(x, y)
            if (e+1)%50 == 0:
                model.save('./Model/{}-{}-{}.pt'.format(project, self.modelType, e+1))
            print(loss)
        return model