import torch
import torch.nn as nn

class Embedders:
    def __init__(self):
        pass

    def getEmbedder(self, embeddingType, embeddingSize):
        return EmbeddingLayer(embeddingSize)

class EmbeddingLayer:
    def __init__(self, embeddingSize):
        self.wordSet = None
        self.embeddingSize = embeddingSize

    def setWordSet(self, wordSet):
        self.wordSet = wordSet
        self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=self.embeddingSize)

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    try:
                        idx = torch.tensor(self.wordSet.index(word))
                        temp.append(idx)
                    except ValueError:
                        continue
                except IndexError:
                    continue
            if temp and len(temp)>3:
                temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords