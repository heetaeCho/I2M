import torch
import torch.nn as nn
import torch.nn.functional as F

class Models:
    @staticmethod
    def getModel(modelType, numClass, numCell, dropout, maxLen):
        if modelType == 'rnn':
            return LSTM(numClass, numCell, dropout)
        elif modelType == 'cnn':
            return CNN(maxLen, numClass, numCell, dropout)

class LSTM(nn.Module):
    def __init__(self, outputSize, inputSize, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=inputSize, \
            batch_first=True, bidirectional=False)
        self.out = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out, (hid, cell) = self.lstm(x)
        out = hid[0]
        out = self.out(out)
        return F.log_softmax(out)

class CNN(nn.Module):
    def __init__(self, maxLen, numClass, numCell, dropout):
        super(CNN, self).__init__()
        self.conv2d_filter2 = nn.Conv2d(1, 64, (2, numCell))
        self.conv2d_filter3 = nn.Conv2d(1, 64, (3, numCell))
        self.conv2d_filter4 = nn.Conv2d(1, 64, (4, numCell))
        self.conv2d_filter5 = nn.Conv2d(1, 64, (5, numCell))

        self.maxpool_filter2 = nn.MaxPool1d(maxLen - 2 + 1)
        self.maxpool_filter3 = nn.MaxPool1d(maxLen - 3 + 1)
        self.maxpool_filter4 = nn.MaxPool1d(maxLen - 4 + 1)
        self.maxpool_filter5 = nn.MaxPool1d(maxLen - 5 + 1)
        self.out = nn.Linear(64 * 4, numClass)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(1), x.size(2))
        out2 = self.conv2d_filter2(x)
        out3 = self.conv2d_filter3(x)
        out4 = self.conv2d_filter4(x)
        out5 = self.conv2d_filter5(x)

        out2 = torch.squeeze(out2)
        out3 = torch.squeeze(out3)
        out4 = torch.squeeze(out4)
        out5 = torch.squeeze(out5)

        out2 = self.maxpool_filter2(out2)
        out3 = self.maxpool_filter3(out3)
        out4 = self.maxpool_filter4(out4)
        out5 = self.maxpool_filter5(out5)

        out2 = out2.view(x.size()[0], -1)
        out3 = out3.view(x.size()[0], -1)
        out4 = out4.view(x.size()[0], -1)
        out5 = out5.view(x.size()[0], -1)

        out = torch.cat((out2, out3, out4, out5), dim=1)
        out = self.out(F.relu(out))
        return F.log_softmax(out)