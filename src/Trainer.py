import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Model import Model

class Trainer:
    def __init__(self, project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType):
        self.project = project
        self.modelPath = classifierPath
        self.modelType = modelType
        self.epoch = epoch
        self.numCell = numCell
        self.batchSize = batchSize
        self.maxLen = maxLen
        self.numClass = numClass
        self.embeddingType = embeddingType
        self.model = Model(modelType, numClass, numCell, maxLen)

    def fit(self, X, Y):
        ## ???
        # X = X.cuda()
        # Y = Y.cuda()
        # x = torch.flip(x, dims=[1]) <<
        # RuntimeError: CUDA error: unspecified launch failure
        # CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
        # For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
        ## ???
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False)

        for e in range(1, self.epoch+1):
            print('{}/{} '.format(e, self.epoch), end='\r')
            _last = False
            for idx, samples in enumerate(loader):
                if idx+1 == len(loader):
                    _last = True
                x, y = samples
                if self.modelType == 'rnn':
                    x = torch.flip(x, dims=[1])
                y = y.view(-1)
                x = x.cuda()
                y = y.cuda()
                self.model.fit(x, y, _last)
            if e%50 == 0:
                self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
        print()