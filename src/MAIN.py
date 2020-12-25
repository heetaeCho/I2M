import os
import numpy as np
from collections import Counter
import json
import time
import torch

from manualData import Manual
from issueData import Issue
from DataReader import DataReader
from Parser import Parser
from TextPreprocessor import TextPreprocessor
from WordEmbedder import WordEmbedder
from Trainer import Trainer
from Classifier import Classifier
from Evaluator import Evaluator
from Logger import Logger

class MAIN:
    def __init__(self):
        pass

    def run(self, informationJson):
        logger = Logger('./Log/log_{}.txt'.format('-'.join(time.ctime().replace(':',';').split(' '))))
        projects = informationJson["projects"]
        dataPath = informationJson["dataPath"]
        embeddingModelPath = informationJson["embeddingModelPath"]
        classifierPath = informationJson["classifierPath"]
        resultPath = informationJson["resultPath"]

        dataTypes = informationJson["dataTypes"]
        embeddingTypes = informationJson["embeddingTypes"]
        embeddingSize = informationJson["embeddingSize"]
        modelTypes = informationJson["modelTypes"]
        epoch = informationJson["epoch"]
        numCell = informationJson["numCell"]
        batchSize = informationJson["batchSize"]

        um = dataTypes[0]
        ir = dataTypes[1]

        for project in projects:
            print('-------------------{}-------------------'.format(project))
            logger.log('\n-------------------{}-------------------\n'.format(project))
            self.manuals = []
            self.issues = []

            self._readData('{}{}/{}/'.format(dataPath, um, project), um)
            self._readData('{}{}/{}/'.format(dataPath, ir, project), ir)
            numClass = len(self.manuals)
            logger.log('numClass => {}\n'.format(numClass))

            self._parse(project, um)
            self._parse(project, ir)

            trainWordSet, trainMaxLen = self._preprocess(project, um)
            testWordSet, testMaxLen = self._preprocess(project, ir)
            wordSet = trainWordSet
            wordSet.extend(testWordSet)
            maxLen = max(trainMaxLen, testMaxLen)
            logger.log('numWords => {}\n'.format(len(wordSet)))
            logger.log('maxSeqLen => {}\n'.format(maxLen))

            for modelType in modelTypes:
                print('-------------------{}-------------------'.format(modelType))
                for embeddingType in embeddingTypes:
                    logger.log('-----------{}----{}-----------\n'.format(modelType, embeddingType))
                    logger.log('-------start-at----{}---------\n'.format('-'.join(time.ctime().replace(':',';').split(' '))))
                    self.train(project, um, classifierPath, modelType, embeddingType, embeddingSize,\
                        epoch, numCell, batchSize, wordSet, maxLen, numClass)

            self.test(project, classifierPath, ir, embeddingSize, wordSet, maxLen, resultPath)
            

    def train(self, project, dataType, classifierPath, modelType, embeddingType, embeddingSize,\
        epoch, numCell, batchSize, wordSet, maxLen, numClass):
        self._embedding(project, dataType, embeddingType, embeddingSize, wordSet, maxLen)
        self._train(project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType)

    def test(self, project, classifierPath, dataType, embeddingSize, wordSet, maxLen, resultPath):
        for modelName in os.listdir('{}{}'.format(classifierPath, project)):
            embeddingType = modelName.split('-')[1]
            self._embedding(project, dataType, embeddingType, embeddingSize, wordSet, maxLen)
            self._classify(project, classifierPath, modelName)
            self._evaluate(project, modelName, resultPath)

    def _readData(self, dataPath, dataType):
        dataReader = DataReader(dataPath, dataType)
        numOfFiles = dataReader.getNumberOfFiles()
        
        for i in range(numOfFiles):
            _file, context = dataReader.readData(i)
            if dataType == 'UserManual':
                manual = Manual()
                manual.id = i
                manual.name = _file.lower()
                manual.sentences = context
                self.manuals.append(manual)
            elif dataType == 'IssueReport':
                issue = Issue()
                issue.number = _file
                issue.html = context
                self.issues.append(issue)
                
    def _parse(self, project, dataType):
        parser = Parser(project, dataType)
        if dataType == 'UserManual':
            data = self.manuals
        else:
            data = self.issues
        
        for d in data:
            if dataType == 'UserManual':
                d.sentences = parser.parse(d.sentences)
            else:
                t, l, b = parser.parse(d.html)
                d.title = t
                d.lables = l
                d.body = b

    def _preprocess(self, project, dataType):
        preprocessor = TextPreprocessor(dataType)
        if dataType == 'UserManual':
            data = self.manuals
        else:
            data = self.issues
        
        for d in data:
            if dataType == 'UserManual':
                d.sentences = preprocessor.pp(d.sentences)
            else:
                d.title, _, d.body = preprocessor.pp((d.title, d.lables, d.body))
        return list(sorted(preprocessor.wordSet)), preprocessor.maxLen

    def _embedding(self, project, dataType, embeddingType, embeddingSize, wordSet, maxLen):
        embedder = WordEmbedder(project, embeddingType, embeddingSize, wordSet, maxLen)
        if dataType == 'UserManual':
            data = self.manuals
        else:
            data = self.issues

        for ix, d in enumerate(data):
            if dataType == 'UserManual':
                d.vectors = embedder.embedding(d.sentences)
            else:
                d.titleVectors = embedder.embedding(d.title)
                d.bodyVectors = embedder.embedding(d.body)

    def _train(self, project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType):
        trainer = Trainer(project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType)

        X, Y = None, None
        for ix, manual in enumerate(self.manuals):
            x = manual.vectors
            if x is None:
                continue
            y = torch.tensor(manual.id).repeat(len(x))
            if X is None:
                X = x
                Y = y
            else:
                X = torch.cat((X, x), dim=0)
                Y = torch.cat((Y, y), dim=0)
        trainer.fit(X, Y)

    def _classify(self, project, classifierPath, modelName):
        classifier = Classifier(project, classifierPath, modelName)
        X = None
        for issue in self.issues:
            x = issue.titleVectors
            if x is None:
                continue
            if X is None:
                X = x
            else:
                X = torch.cat((X, x), dim=0)
        prediction = classifier.classify(X)
        prediction = prediction.detach().cpu().numpy()
        
        idx = 0
        for issue in self.issues:
            if issue.titleVectors is None:
                continue
            else:
                issue.titlePredictedClass = prediction[idx]
                idx += 1

    def _evaluate(self, project, modelName, resultPath):
        evaluator = Evaluator(project, modelName, resultPath, self.manuals, self.issues)
        evaluator.evaluate()

if __name__ == '__main__':
    main = MAIN()
    informationJson = json.loads(open('./I2F-Information.json', 'r', encoding='utf-8').read())
    main.run(informationJson)