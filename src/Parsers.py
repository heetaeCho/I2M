import marko
import re
from bs4 import BeautifulSoup
import bs4

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

    def _getArticle(self, soup):
        soup = soup.select('article.article')
        soup.footer.extract()
        soup.aside.extrace()
        return soup.contents[1]

    def parse(self, text):
        context = []
        soup = BeautifulSoup(text, 'html.parser')

        for child in soup.contents:
            if type(child) != bs4.element.Tag:
                continue
            context.append(child.text.strip())
        return '\n'.join(context)

class mdParser:
    def __init__(self, project):
        self.project = project

    def parse(self, text):
        context = []
        if self.project == 'vscode':
            text = text[7:]
        html = marko.convert(text)
        soup = BeautifulSoup(marko.convert(text), 'html.parser')

        for child in soup.contents:
            if type(child) != bs4.element.Tag:
                continue
            context.append(child.text.strip())
        return '\n'.join(context)

class issueParser:
    def __init__(self):
        pass

    def parse(self, text):
        title_block = 'title'
        labels_block = 'a.sidebar-labels-style,box-shadow-nonewidth-full,d-block,IssueLabel,v-align-text-top'
        context_block = 'div.edit-comment-hide > task-lists > table > tbody > tr > td'

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
