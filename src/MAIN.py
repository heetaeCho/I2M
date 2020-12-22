import argparse
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
        print('main.run()')
        um = dataTypes[0]
        ir = dataTypes[1]
        trainData = self._readData('{}{}/{}/'.format(dataPath, um, project))
        testData = self._readData('{}{}/{}/'.format(dataPath, ir, project))
        # print(len(trainData))
        # print(len(testData))

        trainData = self._parse(project, trainData, um)
        testData = self._parse(project, testData, ir)
        # print(len(trainData[0]))
        # print(len(testData[0]))

        trainData = self._preprocess(project, trainData, um)
        wordSet = list(sorted(TextPreprocessor(um).wordSet))
        testData = self._preprocess(project, testData, ir)
        # print(trainData[0])
        # print(testData[0])
        # print(len(wordSet))

        trainData = self._embedding(trainData, embeddingType, embeddingSize)
        testData = self._embedding(testData, embeddingType, embeddingSize)
        # # print(trainData[0])
        # # print(testData[0])

        # model = self._train(trainData, modelType, epoch, numCell, batchSize, dropout)
        # # print(model)

        # predicted = self._classify(model, testData)
        # # print(predicted)

        # result = self._evaluate(testData, predicted)
        # # print(result)

    def _readData(self, dataPath):
        dataReader = DataReader(dataPath)
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
            parsedData.append(parser.parse(data[i]))
        return parsedData

    def _preprocess(self, project, data, dataType):
        processedData = []
        preprocessor = TextPreprocessor(dataType)
        for i in range(len(data)):
            processedData.append(preprocessor.pp(data[i]))
        return processedData

    def _embedding(self, data, embeddingType, embeddingSize):
        embedder = WordEmbedder(embeddingType, embeddingSize)
        embeddedData = []
        for i in range(len(data)):
            embeddedData.append(embedder.embedding(data[i]))
        return embeddedData

    def _train(self, data, modelType, epoch, numCell, batchSize, dropout):
        trainer = Trainer(modelType, numCell, batchSize, dropout)
        model = trainer.fit(epoch, data)
        return model

    def _classify(self, model, data):
        classifier = Classifier()
        res = classifier.classify(model, data)
        return res

    def _evaluate(self, realData, predictedData):
        evaluator = Evaluator()
        res = evaluator.evaluate(realData, predictedData)
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
                                        print('--------------------------{}----------------------------'.format(project))
                                        main.run(project, dataPath, dataTypes, embeddingType, embeddingSize,\
                                            modelType, epoch, numCell, batchSize, dropout)
                                    exit()