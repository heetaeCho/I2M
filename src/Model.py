import torch
import torch.nn as nn
from Models import Models

class Model:
    def __init__(self, modelType, numClass, numCell, dropout, maxLen):
        self.model = Models.getModel(modelType, numClass, numCell, dropout, maxLen).cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
    
    def fit(self, x, y):
        pred = self.model(x)
        self.optimizer.zero_grad()
        loss = self.criterion(pred, y)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        print(loss)

    def predict(self, x):
        return self.model(data)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)