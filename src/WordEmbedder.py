from Embedders import Embedders

class WordEmbedder:
    embedder = None
    def __init__(self, embeddingType, embeddingSize, wordSet, maxLen):
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
        self.wordSet = wordSet
        self.maxLen = maxLen
    
    def embedding(self, data):
        if WordEmbedder.embedder is None:
            embedder = Embedders()
            WordEmbedder.embedder = embedder.getEmbedder(self.embeddingType, self.embeddingSize)
        WordEmbedder.embedder.setWordSet(self.wordSet)
        return WordEmbedder.embedder.embedding(data, self.maxLen)