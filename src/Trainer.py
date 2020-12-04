from Model import Model

class Trainer:
    def __init__(self, modelType, numCell, batchSize, dropout):
        self.modelType = modelType
        self.numCell = numCell
        self.batchSize = batchSize
        self.dropout = dropout

    def fit(self, epoch, data):
        model = Model(self.modelType, self.numCell, self.batchSize, self.dropout)
        model.fit(epoch, data)
        return model