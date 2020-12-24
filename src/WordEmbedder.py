import torch
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
        
        embeddedData = None
        trainY = None
        for i in range(len(data)):
            embedded = WordEmbedder.embedder.embedding(data[i], self.maxLen)
            if embedded is None:
                continue
            if i == 0:
                embeddedData = embedded
                trainY = torch.tensor(i).repeat(embedded.size(0))
            else:
                embeddedData = torch.cat((embeddedData, embedded))
                trainY = torch.cat((trainY, torch.tensor(i).repeat(embedded.size(0))))
        return embeddedData, trainY