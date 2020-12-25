import torch

class Classifier:
    def __init__(self):
        pass

    def classify(self, modelType, model, data):
        if modelType=='rnn':
            data = torch.flip(data, dims=[1]).cuda()
        else:
            data = data.cuda()
        
        res = model.predict(data)

        res = torch.argmax(res, dim=1)
        return res