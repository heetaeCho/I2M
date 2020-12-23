import re
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

class Parsers:
    def __init__(self):
        pass

    @classmethod
    def getParser(cls, project, dataType):
        if dataType == 'IssueReport':
            return issueParser()
        else:
            if project == 'npp':
                return mdParser(project)
            elif project == 'komodo':
                return komodoParser()
            elif project == 'vscode':
                return mdParser(project)

class komodoParser:
    def __init__(self):
        pass

    def parse(self, text):
        cleaner = Cleaner(style=True, scripts=True)
        text = cleaner.clean_html(text)
        context = []
        lines = text.split('\n')
        sub_pattern = r'\<[^)]*\>'  # ex) '<이게 뭐든>'
        for line in lines:
            line = re.sub(pattern=sub_pattern, repl='', string=line)
            if len(line.strip()) > 0:
                context.append(line)
        return '\n'.join(context)

class mdParser:
    def __init__(self, project):
        self.project = project

    def parse(self, text):
        context = []
        lines = text.split('\n')
        if self.project == 'vscode':
            lines = lines[9:]

        sub_pattern1 = r'\([^)]*\)'  # ex) '(이게 뭐든)'
        sub_pattern2 = r'\<[^)]*\>'  # ex) '<이게 뭐든>'
        # url_pattern = r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'  # ex) 'https://www.google.com'
        for line in lines:
            line = line.strip()
            if line.startswith('!') or len(line) < 1:
                continue
            line = re.sub(pattern=sub_pattern1, repl='', string=line)
            line = re.sub(pattern=sub_pattern2, repl='', string=line)
            line = line.replace('[', '').replace(']', '').replace('*', '').replace('|', '')
            if len(line) > 0:
                context.append(line)
        return '\n'.join(context)

class issueParser:
    def __init__(self):
        pass

    def parse(self, text):
        title_block = 'title'
        labels_block = 'a.sidebar-labels-style,box-shadow-nonewidth-full,d-block,IssueLabel,v-align-text-top'
        context_block = 'div.edit-comment-hide > task-lists > table > tbody > tr > td.d-block.comment-body.markdown-body.js-comment-body'

        soup = BeautifulSoup(text, 'html.parser')

        title = soup.select_one(title_block).text.split('·')[0].strip()
        temp = [x.string for x in soup.select(labels_block)]
        labels = ' '.join(temp) if temp else None

        try:
            temp = [re.sub('(<.*?>)', '', str(x)).strip().split('\n') for x in soup.select_one(context_block).select('p')]
            context = []
            for x in temp:
                while len(x) > 0:
                    context.append(x.pop())
            context = '\n'.join(context)
        except AttributeError:
            context = ''
        return title, labels, context