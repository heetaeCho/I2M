from Crawler import Crawler
import os

class DataReader:
    def __init__(self, dataPath, dataType, project):
        self.dataPath = dataPath
        self.dataType = dataType
        self.project = project
        self.manual_files = []
        self.filters = ['faq', 'license', 'video', 'overview', 'trouble', 'video']
        if project == 'komodo':
            extention = '.html'
        else:
            extention = '.md'
        path = '/'.join(dataPath.split('/')[:-2])+'/'
        folder = dataPath.split('/')[-2]
        self._getManuals(path, folder, extention)
        self.manual_files = sorted(self.manual_files)

    def _crawl(self):
        crawler = Crawler()
        return '_crawl'

    def _readFile(self, i):
        if self.dataType == 'UserManual':
            _file = self.manual_files[i]
            return _file, open(self.dataPath+'/{}'.format(_file), 'r', encoding='utf-8').read()
        else:
            _file = [file for file in sorted(os.listdir(self.dataPath)) if file.endswith('.html')][i]
            return _file.replace('.html', ''), open(self.dataPath+'/{}'.format(_file), 'r', encoding='utf-8').read()

    def _isURL(self):
        if self.dataPath.startswith('http'):
            return True
        return False
    
    def _getManuals(self, path=None, folder=None, extention=None):
        if path is None:
            folder = self.project
            path = self.path+folder+'/'
            extention = self.extention
        else:
            path = path+folder+'/'
        for elem in os.listdir(path):
            isFilter = False
            for filter in self.filters:
                if filter in elem or filter in path:
                    isFilter = True
                    break
            if isFilter: continue
            if os.path.isdir(path+elem):
                self._getManuals(path, elem, extention)
            elif elem.endswith(extention):
                if self.project == 'vscode':
                    elem = path.split('/')[4]+'/'+elem
                self.manual_files.append(elem)

    def getNumberOfFiles(self):
        if self._isURL():
            raise ValueError('it works for only folder')
        else:
            if self.dataType == 'UserManual':
                return len(self.manual_files)
            return len([file for file in sorted(os.listdir(self.dataPath)) if os.path.isfile(self.dataPath+file)])
    
    def readData(self, i):
        if self._isURL():
            return self._crawl()
        else:
            return self._readFile(i)