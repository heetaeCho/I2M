class Logger:
    def __init__(self, logPath):
        self.logger = open(logPath, 'a', encoding='utf-8')