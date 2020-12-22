class Logger:
    def __init__(self, logPath):
        self.logger = open(logPath, 'a', encoding='utf-8')

    def log(self, data):
        self.logger.write(data)
    
    def __del__(self):
        self.logger.close()