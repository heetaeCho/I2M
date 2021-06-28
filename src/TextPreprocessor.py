import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self, dataType):
        self.wordSet = set()
        self.maxLen = 0
        self.dataType = dataType
        try:
            self.stopwords = list(set(stopwords.words('english')))
        except LookupError:
            nltk.download('stopwords')
        finally:
            self.stopwords = list(set(stopwords.words('english')))
        self.lemmatizer = WordNetLemmatizer()

    def pp(self, data):
        return self._preprocess(data)

    def _preprocess(self, data):
        processed = []
        if self.dataType == 'UserManual':
            data = data.lower()
            lines = data.split('\n')
            for line in lines:
                tokens = self._run(line)
                if tokens and len(tokens) >= 3:
                    if len(tokens) > self.maxLen:
                        self.maxLen = len(tokens)
                    processed.append(self._run(line))
                else:
                    continue
            return processed
        else:
            title, _, body = data
            title = [self._run(title.lower())]
            processedBody = []
            if body is not None:
                for line in body.split('\n'):
                    tokens = self._run(line.lower())
                    if tokens and len(tokens) >= 3:
                        processedBody.append(tokens)
            return title, _, processedBody

    def _run(self, line):
        # line = self._clean_text(line)
        tokens = self._tokenize(line)
        tokens = self._stopwordsRemoval(tokens)
        tokens = self._posTag(tokens)
        tokens = self._tagFilter(tokens)
        tokens = self._lemmatize(tokens)
        return tokens

    def _clean_text(self, text):
        text = text.replace('.', ' ').strip()
        text = text.replace("·", " ").strip()
        pattern = '[^ 0-9|a-zA-Z]+'
        text = re.sub(pattern=pattern, repl=' ', string=text)
        return text

    def _tokenize(self, sentence):
        try:
            tokenizer = nltk.word_tokenize
        except LookupError:
            nltk.download('punkt')
        finally:
            tokenizer = nltk.word_tokenize
        return tokenizer(sentence)

    def _stopwordsRemoval(self, words):
        return [word for word in words if word not in self.stopwords]

    def _posTag(self, tokens):
        try:
            tagger = nltk.pos_tag
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')
        finally:
            tagger = nltk.pos_tag
        return tagger(tokens, tagset='universal')

    def _tagFilter(self, tagged):
        # filtered = []
        # for word, pos in tagged:
        #     if pos == '.' or pos == 'NUM':
        #         continue
        #     elif (word == '|') or (word == '>') or (word == '<') or ("'" in word) or ("/" in word):
        #         continue
        #     else:
        #         filtered.append(word)
        # return filtered
        filtered = []
        for word, pos in tagged:
            if pos == '.' or pos == 'Num':
                continue 
            elif not word.isalpha():
                continue
            else:
                filtered.append(word)
        return filtered

    def _lemmatize(self, tokens):
        lemedTokens = []
        for token in tokens:
            # if token != 'vs':
            #     token = self.lemmatizer.lemmatize(token)
            lemedTokens.append(token)
            if token not in self.wordSet:
                self.wordSet.add(token)
        return lemedTokens