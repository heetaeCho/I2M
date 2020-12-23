from Embedders import Embedders

class WordEmbedder:
    embedder = None
    def __init__(self, project, embeddingType, embeddingSize, wordSet, maxLen):
        self.project = project
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
        self.wordSet = wordSet
        self.maxLen = maxLen
    
    def embedding(self, data):
        embedder = Embedders()
        Embedders.wordSet = self.wordSet
        WordEmbedder.embedder = embedder.getEmbedder(self.project, self.embeddingType, self.embeddingSize)
        
        embeddedData = []
        for i in range(len(data)):
            embeddedData.append(WordEmbedder.embedder.embedding(data[i], self.maxLen))
        return embeddedData