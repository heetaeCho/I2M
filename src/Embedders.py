class Embedders:
    def __init__(self):
        pass

    @classmethod
    def getEmbedder(cls, embeddingType, embeddingSize):
        return "{}".format(embeddingType)