import os
import numpy as np
from collections import Counter
import json
import time
import torch
import torch.nn as nn

from TextPreprocessor import TextPreprocessor
from Trainer import Trainer
from Classifier import Classifier
from Logger import Logger

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class EmbeddingLayer:
    def __init__(self, project, wordSet, embeddingSize):
        self.wordSet = wordSet
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer_tt.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer_tt.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            idx = torch.tensor(self.wordSet.index(word))
                            if idx > len(self.wordSet):
                                continue
                            temp.append(idx)
                        except ValueError:
                            continue
                except IndexError:
                    continue
            if len(temp)>1:
                with torch.no_grad():
                    temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class Evaluator:
    def __init__(self, project, resultPath, modelName, task, k):
        self.resultPath = resultPath
        self.modelName = modelName
        self.project = project
        self.task = task
        self.k = k

    def evaluate(self, predicted, real, tp='title'):        
        logger = Logger('{}{}/{}-{}-{}.txt'.format(self.resultPath, self.project,self.modelName, self.task, self.k))
        logger.log('real=> {}\n'.format(str(Counter(real))))
        logger.log('predicted=> {}\n'.format(str(Counter(predicted))))

        precision = precision_score(real, predicted, average='weighted', zero_division=0)
        recall = recall_score(real, predicted, average='weighted', zero_division=0)
        f1 = f1_score(real, predicted, average='weighted', zero_division=0)
        acc = accuracy_score(real, predicted)
        logger.log('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))

        # precision = precision_score(real, predicted, average='micro', zero_division=0)
        # recall = recall_score(real, predicted, average='micro', zero_division=0)
        # f1 = f1_score(real, predicted, average='micro', zero_division=0)
        # acc = accuracy_score(real, predicted)
        # logger.log('-------micro-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))

class MAIN:
    def __init__(self):
        pass

    def run(self):
        basePath = './Data/TicketTagger/'
        modelPath = './Model_tt/'
        projects = ['npp', 'komodo', 'vscode']
        modelTypes = ['cnn', 'rnn']
        numTask = 5
        numK = 10
        wordSet = None

        for project in projects:
            if project == 'npp':
                labels = ['__label__bug', '__label__feature', '__label__enhancement']
            elif project == 'komodo':
                labels = ['__label__bug', '__label__enhencement']
            elif project == 'vscode':
                labels = ['__label__bug', '__label__feature']

            for modelType in modelTypes:
                for task in range(1, numTask+1):
                    for k in range(numK):
                        print('----{}----{}----{}----{}----'.format(project, modelType, task, k))
                        train, test = self._readFile(basePath, project, task, k)
                        train, test = self._transform(train, test)
                        trainX, trainY, trainWordSet = self._preprocess(train)
                        testX, testY, testWordSet = self._preprocess(test)
                        if wordSet is None:
                            wordSet = trainWordSet
                            wordSet.extend(testWordSet)
                        trainLen = np.max([len(x[0]) for x in trainX])
                        testLen = np.max([len(x[0]) for x in testX])
                        maxLen = np.max([trainLen, testLen])

                        trainX, trainY = self._embedding(project, trainX, trainY, wordSet, labels, maxLen)
                        testX, testY = self._embedding(project, testX, testY, wordSet, labels, maxLen)

                        self._train(trainX, trainY, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
                        self._test(testX, testY, project, modelPath, modelType, task, k)
            wordSet = None

    def _readFile(self, path, project, task, k):
        trainFile = '{}{}/task{}/train_{}.txt'.format(path, project, task, k)
        testFile = '{}{}/task{}/test_{}.txt'.format(path, project, task, k)
        train = open(trainFile, 'r').read()
        test = open(testFile, 'r').read()
        return train, test
    
    def _transform(self, train, test):
        train = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in train.split('\n')[:-1]]
        test = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in test.split('\n')[:-1]]
        return train, test

    def _preprocess(self, data):
        preprocessor = TextPreprocessor('tt')
        x = []
        y = []
        for d in data:
            t, l, _ = preprocessor.pp(d)
            if t:
                x.append(t)
                y.append(l)
            else:
                continue
        return x, y, list(preprocessor.wordSet)

    def _embedding(self, project, x, y, wordSet, labels, maxLen):
        embedder = EmbeddingLayer(project, wordSet, 300)
        X = None
        Y = None
        for t, l in zip(x, y):
            emWords = embedder.embedding(t, maxLen)
            l = torch.tensor(labels.index(l)).view(-1)
            if emWords is not None:
                if X is None:
                    X = emWords
                    Y = l
                else:
                    X = torch.cat((X, emWords), dim=0)
                    Y = torch.cat((Y, l), dim=0)
            else:
                continue
        return X, Y

    def _train(self, X, Y, project, modelPath, modelType, epoch, batchSize, maxLen, numClass):
        trainer = Trainer(project, modelPath, modelType, epoch, 300, batchSize, maxLen, numClass, 'EmbeddingLayer')
        trainer.fit(X, Y)

    def _test(self, X, Y, project, modelPath, modelType, task, k):
        Y = Y.detach().cpu().numpy()
        for modelName in os.listdir(modelPath+project):
            classifier = Classifier(project, modelPath, modelName)
            prediction = classifier.classify(X).detach().cpu().numpy()
            self._evaluate(prediction, Y, project, modelName, task, k)

    def _evaluate(self, prediction, real, project, modelName, task, k):
        evaluator = Evaluator(project, './Result_tt/', modelName, task, k)
        evaluator.evaluate(prediction, real)

if __name__ == '__main__':
    main = MAIN()
    main.run()