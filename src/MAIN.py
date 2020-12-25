import os
import argparse
import numpy as np
from collections import Counter
import json

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

    def run(self, project, dataPath, dataTypes, embeddingType, embeddingSize,\
            modelType, epoch, numCell, batchSize, dropout):
        print("---------main.run()---------{}---------".format(project))
        um = dataTypes[0]
        ir = dataTypes[1]
        trainData = self._readData('{}{}/{}/'.format(dataPath, um, project), um)
        testData = self._readData('{}{}/{}/'.format(dataPath, ir, project), ir)

        numClass = len(trainData)
        issueNumbers = list(np.asarray(testData)[:, 0])
        testData = list(np.asarray(testData)[:, 1])

        trainData = self._parse(project, trainData, um)
        testData = self._parse(project, testData, ir)

        trainData, trainWordSet, trainMaxLen = self._preprocess(project, trainData, um)
        testData, testWordSet, testMaxLen = self._preprocess(project, testData, ir)
        wordSet = trainWordSet
        wordSet.extend(testWordSet)
        maxLen = max(trainMaxLen, testMaxLen)
        
        self.train(project, trainData, wordSet, maxLen, numClass, modelType, epoch, numCell, batchSize, dropout)
        self.test(project, testData, wordSet, maxLen, modelType)

    def train(self, project, trainData, wordSet, maxLen, numClass, modelType, epoch, numCell, batchSize, dropout):
        trainData, trainY = self._embedding(project, trainData, embeddingType, embeddingSize, wordSet, maxLen)
        self._train(project, trainData, trainY, numClass, modelType, epoch, numCell, batchSize, dropout, maxLen)

    def test(self,  project, testData, wordSet, maxLen, modelType):
        testDataTitle = np.asarray(testData)[:, 0].reshape(1, -1)
        testDataLables = np.asarray(testData)[:, 1]
        testDataBody = np.asarray(testData)[:, 2]

        testDataTitle, _ = self._embedding(project, testDataTitle, embeddingType, embeddingSize, wordSet, maxLen)
        testDataBody, _ = self._embedding(project, testDataBody, embeddingType, embeddingSize, wordSet, maxLen)

        for model in os.listdir('./Model/{}'.format(project)):
            res = self._classify(project, modelType, model, testDataTitle)
            print(res)

    def _readData(self, dataPath, dataType):
        dataReader = DataReader(dataPath, dataType)
        numOfFiles = dataReader.getNumberOfFiles()
        data = []
        # for i in range(numOfFiles):
        for i in range(2):
            data.append(dataReader.readData(i))
        return data

    def _parse(self, project, data, dataType):
        parser = Parser(project, dataType)
        parsedData = []
        for i in range(len(data)):
            try:
                parsedData.append(parser.parse(data[i]))
            except AttributeError:
                print(i)
                print(data[i])
        return parsedData

    def _preprocess(self, project, data, dataType):
        processedData = []
        preprocessor = TextPreprocessor(dataType)
        for i in range(len(data)):
            processedData.append(preprocessor.pp(data[i]))
        return processedData, list(sorted(preprocessor.wordSet)), preprocessor.maxLen

    def _embedding(self, project, data, embeddingType, embeddingSize, wordSet, maxLen):
        embedder = WordEmbedder(project, embeddingType, embeddingSize, wordSet, maxLen)
        embeddedData, trainY = embedder.embedding(data)
        return embeddedData, trainY

    def _train(self, project, data, trainY, numClass, modelType, epoch, numCell, batchSize, dropout, maxLen):
        trainer = Trainer(modelType, numCell, batchSize, dropout, maxLen)
        trainer.fit(project, epoch, data, trainY, numClass)

    def _classify(self, project, modelType, modelName, data):
        classifier = Classifier()
        res = classifier.classify(project, modelType, modelName, data)
        return res

    def _evaluate(self, issueNumbers, titlePrediction):
        evaluator = Evaluator()
        res = evaluator.evaluate(issueNumbers, titlePrediction)
        return res

def add_arg2parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='vscode, komodo, npp')
    parser.add_argument('--dataPath', help='URL or Folder')
    # parser.add_argument('--dataType', help='UserManual or IssueReport')
    parser.add_argument('--embeddingType', help='EmbeddingLayer, W2V, Glove, FastText')
    parser.add_argument('--embeddingSize', help='shold be int N', type=int)
    parser.add_argument('--modelType', help='cnn or rnn')
    parser.add_argument('--epoch', help='should be int N', type=int)
    parser.add_argument('--numCell', help='should be int N', type=int)
    parser.add_argument('--batchSize', help='should be int N', type=int)
    parser.add_argument('--dropout', help='should be float 0.xx', type=float)
    return parser
    
def checkArgs(args):
    values = Counter(args.__dict__.values())
    if values[None] > 2:
        return False
    elif values[None] == 2:
        if (args.numCells == None and args.modelType == 'cnn') \
            and (args.embeddingSize == None and args.embeddingType != 'EmbeddingLayer'):
            return True
    elif values[None] == 1:
        if (args.numCells == None and args.modelType == 'cnn') \
            or (args.embeddingSize == None and args.embeddingType != 'EmbeddingLayer'):
            return True
        else:
            return False
    else:
        return False

if __name__ == '__main__':
    parser = add_arg2parser()
    args = parser.parse_args()
    main = MAIN()

    if checkArgs(args):
        project = args.project
        dataPath = args.dataPath
        dataTypes = ["UserManual", "IssueReport"]
        embeddingType = args.embeddingType
        embeddingSize = args.embeddingSize
        modelType = args.modelType
        epoch = args.epoch
        numCell = args.numCell
        batchSize = args.batchSize
        dropout = args.dropout
        main.run(project, dataPath, dataTypes, embeddingType, embeddingSize,\
                modelType, epoch, numCell, batchSize, dropout)
    else:
        informationJson = json.loads(open('./I2F-Information.json', 'r', encoding='utf-8').read())
        projects = informationJson["projects"]
        dataPath = informationJson["dataPath"]
        dataTypes = informationJson["dataTypes"]
        embeddingTypes = informationJson["embeddingTypes"]
        embeddingSizes = informationJson["embeddingSizes"]
        modelTypes = informationJson["modelTypes"]
        epochs = informationJson["epochs"]
        numCells = informationJson["numCells"]
        batchSizes = informationJson["batchSizes"]
        dropouts = informationJson["dropouts"]

        for dropout in dropouts:
            for batchSize in batchSizes:
                for numCell in numCells:
                    for epoch in epochs:
                        for modelType in modelTypes:
                            for embeddingSize in embeddingSizes:
                                for embeddingType in embeddingTypes:
                                    WordEmbedder.embedder = None
                                    for project in projects:
                                        print('----------------------{}-{}------------------------'.format(project, embeddingType))
                                        main.run(project, dataPath, dataTypes, embeddingType, embeddingSize,\
                                            modelType, epoch, numCell, batchSize, dropout)
                                    exit()