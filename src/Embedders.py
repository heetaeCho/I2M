import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText as FT
from gensim.scripts.glove2word2vec import glove2word2vec

def convertModel():
    glvFile = './EmbeddingModel/GloVe.txt'
    tmpFile = './EmbeddingModel/GloVe-W2V.pt'
    try:
        return KeyedVectors.load_word2vec_format(tmpFile)
    except FileNotFoundError:
        glove2word2vec(glvFile, tmpFile)
        return KeyedVectors.load_word2vec_format(tmpFile)

class Embedders:
    def __init__(self):
        self._wordSet = None

    @property
    def wordSet(self):
        return self._wordSet

    @wordSet.setter
    def wordSet(self, wordSet):
        self._wordSet = wordSet

    def embedding(self):
        raise ValueError

    def getEmbedder(self, project, embeddingType, embeddingSize):
        if embeddingType == "EmbeddingLayer":
            return EmbeddingLayer(project, embeddingSize)
        elif embeddingType == "W2V":
            return W2V(project, embeddingSize)
        elif embeddingType == "Glove":
            return GloVe(project, embeddingSize)
        elif embeddingType == "FastText":
            return FastText(project, embeddingSize)

class EmbeddingLayer(Embedders):
    def __init__(self, project, embeddingSize):
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(super().wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            idx = torch.tensor(super().wordSet.index(word))
                            temp.append(idx)
                        except ValueError:
                            continue
                except IndexError:
                    continue
            if len(temp)>3:
                with torch.no_grad():
                    temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class W2V(Embedders):
    def __init__(self, project, embeddingSize):
        self.w2v = KeyedVectors.load_word2vec_format('./EmbeddingModel/Word2Vec.bin', binary=True)
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(super().wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = None
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            res = torch.from_numpy(self.w2v[word]).view(1, -1)
                        except KeyError:
                            res = self.embedder(torch.tensor(super().wordSet.index(word))).view(1, -1)
                    if temp is None:
                        temp = res
                    else:
                        temp = torch.cat((temp, res), axis=0)
                except IndexError:
                    continue
            if temp is None:
                return None
            if len(temp) > 3:
                temp = torch.cat((temp, torch.zeros(maxLen-len(temp), self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class GloVe(Embedders):
    def __init__(self, project, embeddingSize):
        self.gv = convertModel()
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(super().wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = None
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            res = torch.from_numpy(self.gv[word]).view(1, -1)
                        except KeyError:
                            res = self.embedder(torch.tensor(super().wordSet.index(word))).view(1, -1)
                    if temp is None:
                        temp = res
                    else:
                        temp = torch.cat((temp, res), axis=0)
                except IndexError:
                    continue
            if temp is None:
                return None
            if len(temp) > 3:
                temp = torch.cat((temp, torch.zeros(maxLen-len(temp), self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class FastText(Embedders):
    def __init__(self, project, embeddingSize):
        self.ft = FT.load_fasttext_format('./EmbeddingModel/fastText.bin')
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(super().wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for idx, line in enumerate(lines):
            temp = None
            for i in range(maxLen):
                try:
                    word = line[i]
                    try:
                        res = torch.from_numpy(self.ft.wv[word]).view(1, -1)
                    except KeyError:
                        res = self.embedder(torch.tensor(super().wordSet.index(word))).view(1, -1)
                    if temp is None:
                        temp = res
                    else:
                        temp = torch.cat((temp, res), axis=0)
                except IndexError:
                    continue
            if temp is None:
                return None
            if len(temp) > 3:
                temp = torch.cat((temp, torch.zeros(maxLen-len(temp), self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords