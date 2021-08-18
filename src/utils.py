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
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

        self.dict = {}
        self.dict['SOS'] = 0       
        self.dict['EOS'] = 1     
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i + 2            

    def encode(self, text):
        """对target_label做编码和对齐
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor targets:max_length × batch_size
        """
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
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        texts = list(self.dict.keys())[list(self.dict.values()).index(t)]
        return texts

def loadData(v, data):
    with torch.no_grad():
      v.resize_(data.size()).copy_(data)

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

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