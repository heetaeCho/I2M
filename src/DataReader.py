from Crawler import Crawler
import os

class DataReader:
    def __init__(self, dataPath, dataType):
        self.dataPath = dataPath
        self.dataType = dataType

    def _crawl(self):
        crawler = Crawler()
        return '_crawl'

    def _readFile(self, i):
        if self.dataType == 'UserManual':
            _file = sorted(os.listdir(self.dataPath))[i]
            return _file.split('.')[0], open(self.dataPath+'/{}'.format(_file), 'r', encoding='utf-8').read()
        else:
            _file = sorted(os.listdir(self.dataPath))[i]
            return _file.replace('.txt', ''), open(self.dataPath+'/{}'.format(_file), 'r', encoding='utf-8').read()

    def _isURL(self):
        if self.dataPath.startswith('http'):
            return True
        return False
    
    def getNumberOfFiles(self):
        if self._isURL():
            raise ValueError('it works for only folder')
        else:
            return len(os.listdir(self.dataPath))
    
    def readData(self, i):
        if self._isURL():
            return self._crawl()
        else:
            return self._readFile(i)