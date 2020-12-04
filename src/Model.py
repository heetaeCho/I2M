from Models import Models

class Model:
    def __init__(self, modelType, numCell, batchSize, dropout):
        self.model = Models.getModel(modelType, numCell, batchSize, dropout)
    
    def fit(self, epoch, data):
        return self.model.fit(epoch, data)

    def predict(self, data):
        return self.model.predict(data)