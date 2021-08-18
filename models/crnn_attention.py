# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

GO = 0
EOS_TOKEN = 1             

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.cat_for_back = nn.Linear(nHidden * 2, nHidden)

    def forward(self, input):
        # input (len sequence, batch_size, input size)
        out, _ = self.rnn(input) # recurrent ( T = len sequence, b =  batch size, h = hidden size * 2)
        T, b, h = out.size() 
        t_rec = out.view(T * b, h) #(len sequence * batch size, hidden size * 2)
        out = self.cat_for_back(t_rec)  # (len sequence * batch size, output size)
        out = out.view(T, b, -1) # (len sequence, batch size, output size)

        return out  

class CNN(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''
    def __init__(self, imgH, nc, nh, leakyRelu=False):
        super(CNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=True):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 2), (0, 0)))  # 256x4x16
        convRelu(4)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 0)))  # 512x2x16
        convRelu(6)  # 512x1x16

        self.cnn = cnn

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh)
            )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        # print(b,c,h,w)
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # (b,c,w)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        encoder_outputs = self.rnn(conv)          # seq * batch * n_classes

        return encoder_outputs

class AttentionDec(nn.Module):
    def __init__(self, output_size, hidden_size, drop_value=0.1):
        super(AttentionDec, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_value = drop_value

        self.emb = nn.Embedding(self.output_size, self.hidden_size)
        self.drop = nn.Dropout(self.drop_value)
        self.i2h = nn.Linear(self.hidden_size, self.hidden_size,bias=False)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.score = nn.Linear(self.hidden_size, 1, bias=False)
        self.rnn = nn.GRU(hidden_size*2, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, prev_hidden, enc_outputs ): 
        cur_embeddings = self.emb(input)
        cur_embeddings = self.drop(cur_embeddings) #[32, 256]

        nT = enc_outputs.size(0)
        nB = enc_outputs.size(1)
        nC = enc_outputs.size(2)
        hidden_size = self.hidden_size

        enc_outputs_proj = self.i2h(enc_outputs.view(-1,nC)) # enc_outputs (nT*nB, nC), enc_outputs_proj (nT*nB, hidden_size)
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(enc_outputs_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1) # (nB, nT)
        alpha = F.softmax(emition, dim = 1 ) # nB * nT
        context = (enc_outputs * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0) # (nB, nC) [32,256]
        context = torch.cat([context, cur_embeddings], 1).unsqueeze(0)#(batch_size, output size) 
        out, hid = self.rnn(context, prev_hidden)
        out = F.log_softmax(self.out(out[0]), dim =1)
        return out, hid, alpha

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result