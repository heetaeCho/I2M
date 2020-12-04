class Classifier:
    def __init__(self):
        pass

    def classify(self, model, data):
        res = model.predict(data)
        return res