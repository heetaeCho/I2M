from Embedders import Embedders

class WordEmbedder:
    embedder = None
    def __init__(self, embeddingType, embeddingSize):
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
    
    def embedding(self, data):
        if WordEmbedder.embedder is None:
            WordEmbedder.embedder = Embedders.getEmbedder(self.embeddingType, self.embeddingSize)
        return "WordEmbedder.embedder.embedding(data)"