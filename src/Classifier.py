import torch
import os
from Model import Model

class Classifier:
    def __init__(self):
        pass

    def classify(self, project, modelType, modelName, data):
        model = Model(test=True)
        model.load('./Model/{}/{}'.format(project, modelName))
        if modelType=='rnn':
            data = torch.flip(data, dims=[1]).cuda()
        else:
            data = data.cuda()
        prediction = torch.argmax(model.predict(data), dim=1)
        return prediction