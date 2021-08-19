import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

with open('/home/ngoc/ml/crnn/tts/InkData_line_processed/char') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)       
f.close()

class strLabelConverterForAttention(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS'] = 0       
        self.dict['EOS'] = 1     
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 2            

    def encode(self, text):
        if isinstance(text, str):
            text = [self.dict[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]          

            max_length = max([len(x) for x in text])        
            nb = len(text)
            targets = torch.ones(nb, max_length + 2) * 2              # use ‘blank’ for pading
            for i in range(nb):
                targets[i][0] = 0                          
                targets[i][1:len(text[i]) + 1] = text[i]
                targets[i][len(text[i]) + 1] = 1
            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

def loadData(v, data):
    with torch.no_grad():
      v.resize_(data.size()).copy_(data)

class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def weights_init(model):
    # Official init from torch repo.
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)