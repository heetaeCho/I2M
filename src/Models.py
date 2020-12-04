class Models:
    @staticmethod
    def getModel(modelType, numCell, batchSize, dropout):
        return LSTM()

class LSTM:
    def __init__(self):
        pass

    def fit(self, epoch, data):
        return "model.fit(epoch, data)"

    def predict(self, data):
        return "model.predict(data)"