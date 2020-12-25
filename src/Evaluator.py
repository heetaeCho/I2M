from collections import Counter
import pandas as pd
from Logger import Logger

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self, project, modelName, resultPath, manuals, issues):
        self.project = project
        self.manuals = manuals
        self.issues = issues
        self.logger = Logger('{}{}-{}.txt'.format(resultPath, project, modelName.split('.')[0]))
        self._prepare()

    def _prepare(self):
        docs = [manual.name for manual in self.manuals]
        ansPath = './AnswerSet/'
        df = pd.read_csv(ansPath+self.project+'.csv', encoding='cp949')
        df = df[['Document Files', 'Issue Number']]
        for issue in self.issues:
            idx = df[df['Issue Number'] == int(issue.number)].index
            doc = df.iloc[idx]['Document Files'].values[0].split('.')[0].lower()
            issue.realClass = docs.index(doc)

    def evaluate(self):
        real = []
        predicted = []
        for issue in self.issues:
            if issue.titleVectors is not None:
                real.append(issue.realClass)    
                predicted.append(issue.titlePredictedClass)

        self.logger.log('real=> {}\n'.format(str(Counter(real))))
        self.logger.log('predicted=> {}\n'.format(str(Counter(predicted))))

        precision = precision_score(real, predicted, average='weighted', zero_division=0)
        recall = recall_score(real, predicted, average='weighted', zero_division=0)
        f1 = f1_score(real, predicted, average='weighted', zero_division=0)
        self.logger.log('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\n'.format(precision, recall, f1))

        precision = precision_score(real, predicted, average='micro', zero_division=0)
        recall = recall_score(real, predicted, average='micro', zero_division=0)
        f1 = f1_score(real, predicted, average='micro', zero_division=0)
        self.logger.log('-------micro-------\nprecision: {}\nrecall: {}\nf1-score: {}\n'.format(precision, recall, f1))