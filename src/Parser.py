from Parsers import Parsers

class Parser:
    def __init__(self, project, dataType):
        self.project = project
        self.dataType = dataType

    def parse(self, data):
        return self._parse(data)

    def _parse(self, data):
        parser = Parsers().getParser(self.project, self.dataType)
        return parser.parse(data)