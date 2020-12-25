import torch
import os
from Model import Model

class Classifier:
    def __init__(self, project, classifierPath, modelName):
        self.modelName = modelName
        self.model = Model(test=True)
        self.model.load('{}{}/{}'.format(classifierPath, project, modelName))

    def classify(self, data):
        if self.modelName.startswith('rnn'):
            data = torch.flip(data, dims=[1]).cuda()
        else:
            data = data.cuda()
        prediction = torch.argmax(self.model.predict(data), dim=1)
        return prediction