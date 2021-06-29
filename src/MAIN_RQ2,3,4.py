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
        self.all_manuals = {}
        self.all_issues = {}
        self.max_len = {}
        self.word_set = {}
        self.num_class = {}
        self.embedded_manuals = None

    def run(self, informationJson):
        logger = Logger('./Log/log_{}.txt'.format('-'.join(time.ctime().replace(':',';').split(' '))))
        projects = informationJson["projects"]
        dataPath = informationJson["dataPath"]
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

        for embeddingType in embeddingTypes:
            WordEmbedder.embedder = None
            for project in projects:
                if embeddingType == 'EmbeddingLayer': continue
                if embeddingType == 'W2V' and project == 'npp': continue
                if embeddingType == "EmbeddingLayer":
                    WordEmbedder.embedder = None
                print('-------------------{}-{}------------------'.format(project, embeddingType))
                self.embedded_manuals = None
                if project not in self.all_manuals.keys():
                    self.project = project
                    self.manuals = []
                    self.issues = []

                    self._readData('{}{}/{}/'.format(dataPath, um, project), um)
                    self._readData('{}{}/{}/'.format(dataPath, ir, project), ir)
                    numClass = len(self.manuals)
                    self._parse(project, um)
                    self._parse(project, ir)

                    trainWordSet, trainMaxLen = self._preprocess(project, um)
                    testWordSet, testMaxLen = self._preprocess(project, ir)
                    wordSet = trainWordSet
                    wordSet.extend(testWordSet)
                    # print("total #words=> {}".format(len(wordSet)))
                    # print("intersection #words=> {}".format(len(set(trainWordSet).intersection(set(testWordSet)))))
                    # continue
                    maxLen = max(trainMaxLen, testMaxLen)

                    self.all_manuals[project] = self.manuals
                    self.all_issues[project] = self.issues
                    self.max_len[project] = maxLen
                    self.word_set[project] = wordSet
                    self.num_class[project] = numClass
                else:
                    self.manuals = self.all_manuals[project]
                    self.issues = self.all_issues[project]
                    maxLen = self.max_len[project]
                    wordSet = self.word_set[project]
                    numClass = self.num_class[project]

                logger.log('\n-------------------{}-------------------\n'.format(project))
                logger.log('numClass => {}\n'.format(numClass))
                logger.log('numWords => {}\n'.format(len(wordSet)))
                logger.log('maxSeqLen => {}\n'.format(maxLen))

                for modelType in modelTypes:
                    print('-----------{}----{}-----------'.format(modelType, embeddingType))
                    logger.log('-----------{}----{}-----------\n'.format(modelType, embeddingType))
                    logger.log('-------start-at----{}---------\n'.format('-'.join(time.ctime().replace(':',';').split(' '))))
                    self.train(project, um, classifierPath, modelType, embeddingType, embeddingSize,\
                        epoch, numCell, batchSize, wordSet, maxLen, numClass)

                self.test(project, classifierPath, ir, embeddingSize, wordSet, maxLen, resultPath, numClass, embeddingType)
            # exit()

    def train(self, project, dataType, classifierPath, modelType, embeddingType, embeddingSize,\
        epoch, numCell, batchSize, wordSet, maxLen, numClass):
        print('-------------train-------------')
        if self.embedded_manuals is None:
            self._embedding(project, dataType, embeddingType, embeddingSize, wordSet, maxLen)
            self.embedded_manuals = self.manuals
        else:
            self.manuals = self.embedded_manuals
        self._train(project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType)

    def test(self, project, classifierPath, dataType, embeddingSize, wordSet, maxLen, resultPath, numClass, embeddingType):
        model_names = [model_name for model_name in os.listdir('{}{}'.format(classifierPath, project)) if embeddingType in model_name]
        self._embedding(project, dataType, embeddingType, embeddingSize, wordSet, maxLen)
        for modelName in model_names:
            print('-------test------------{}-----------'.format(modelName))
            embeddingType = modelName.split('-')[1]
            self._classify(project, classifierPath, modelName)
            self._evaluate(project, modelName, resultPath, numClass)

    def _readData(self, dataPath, dataType):
        def findURL(dataPath, file):
            from bs4 import BeautifulSoup
            content = open(dataPath+file, 'r', encoding='utf-8').read()
            soup = BeautifulSoup(content, 'html.parser')
            ref = soup.find('link').get('href')
            return ref.split('/')[-1].lower().replace('.html', '')

        dataReader = DataReader(dataPath, dataType, self.project)
        numOfFiles = dataReader.getNumberOfFiles()
        for i in range(numOfFiles):
            _file, context = dataReader.readData(i)
            if dataType == 'UserManual':
                manual = Manual()
                manual.id = i
                manual.name = _file.split('.')[0].lower()
                if self.project == 'komodo':
                    manual.url = findURL(dataPath, _file)
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

        for idx, d in enumerate(data):
            if dataType == 'UserManual':
                d.vectors = embedder.embedding(d.sentences)
            else:
                d.titleVectors = embedder.embedding(d.title)
                # d.bodyVectors = embedder.embedding(d.body)
            print('embedding progress=> {}/{}'.format(idx+1, len(data)), end='\r')
        print()

    def _train(self, project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType):
        trainer = Trainer(project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType)

        X, Y = None, None
        for manual in self.manuals:
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
        self._classifyTitle(classifier)
        # self._classifyBody(classifier)

    def _classifyTitle(self, classifier):
        # title
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

    def _classifyBody(self, classifier):
        # body
        idx = 0
        predictions = []
        for issue in self.issues:
            x = issue.bodyVectors
            if x is None:
                continue
            prediction = classifier.classify(x, tp='body')
            prediction = prediction.detach().cpu().numpy()
            predIdxs = np.argmax(prediction, axis=1)
            if len(predIdxs) > 2:
                predictions.append(np.bincount(predIdxs).argmax())
            elif len(predIdxs) == 2:
                if predIdxs[0] == predIdxs[1]:
                    predictions.append(predIdxs[0])
                else:
                    p0 = prediction[0][predIdxs[0]]
                    p1 = prediction[1][predIdxs[1]]
                    predictions.append(predIdxs[np.argmax([p0, p1])])
            else:
                predictions.append(predIdxs[0])
        
        idx = 0
        for issue in self.issues:
            if issue.bodyVectors is None:
                continue
            else:
                issue.bodyPredictedClass = predictions[idx]
                idx += 1

    def _evaluate(self, project, modelName, resultPath, numClass):
        evaluator = Evaluator(project, modelName, resultPath, self.manuals, self.issues, numClass)
        evaluator.evaluate(tp='title')
        # evaluator.evaluate(tp='body')

if __name__ == '__main__':
    main = MAIN()
    informationJson = json.loads(open('./I2F-Information.json', 'r', encoding='utf-8').read())
    main.run(informationJson)