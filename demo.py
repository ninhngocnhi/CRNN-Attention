# coding:utf-8
import src.utils as utils
import torch
import cv2
import numpy as np 
from PIL import Image
import torchvision.transforms as transforms
import models.crnn_attention as crnn
from src.utils import alphabet
import matplotlib.pyplot as plt


class attention_ocr():
    def __init__(self):
        encoder_path = '/home/ngoc/ml/CRNN_Attention/expr/encoder_450.pth'
        decoder_path = '/home/ngoc/ml/CRNN_Attention/expr/decoder_450.pth'
        self.max_length = 100                 
        self.EOS_TOKEN = 1
        self.use_gpu = False
        self.max_width = 512
        self.converter = utils.strLabelConverterForAttention(alphabet)
        self.transform = transforms.ToTensor()

        nclass = len(alphabet) + 3
        encoder = crnn.CNN(32, 3, 256)         
        decoder = crnn.AttentionDec(nclass, 256)  

        if encoder_path and decoder_path:
            print('loading pretrained models ......')
            encoder.load_state_dict(torch.load(encoder_path))
            decoder.load_state_dict(torch.load(decoder_path))
        if torch.cuda.is_available() and self.use_gpu:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()

    def constant_pad(self, img_crop):
        h, w, c = img_crop.shape
        ratio = h / 32
        new_w = int(w / ratio)
        new_img = cv2.resize(img_crop,(new_w, 32))
        container = np.ones((32, self.max_width, 3), dtype=np.uint8) * new_img[-3,-3,:]
        if new_w <= self.max_width:
            container[:,:new_w,:] = new_img
        elif new_w > self.max_width:
            container = cv2.resize(new_img, (self.max_width, 32))

        img = Image.fromarray(container.astype('uint8')).convert('RGB')
        img = self.transform(img)
        img.sub_(0.5).div_(0.5)
        if self.use_gpu:
            img = img.cuda()
        return img.unsqueeze(0)
    
    def predict(self, img_crop):
        img_tensor = self.constant_pad(img_crop)
        encoder_out = self.encoder(img_tensor)

        decoded_words = []
        prob = 1.0
        decoder_input = torch.zeros(1).long()      
        decoder_hidden = self.decoder.initHidden(1)
        if torch.cuda.is_available() and self.use_gpu:
            decoder_input = decoder_input.cuda()
            decoder_hidden = decoder_hidden.cuda()
        for di in range(self.max_length): 
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_out)
            probs = torch.exp(decoder_output)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            prob *= probs[:, ni]
            if ni == self.EOS_TOKEN:
                break
            else:
                decoded_words.append(self.converter.decode(ni))

        words = ''.join(decoded_words)
        prob = prob.item()

        return words, prob

if __name__ == '__main__':
    path = '/home/ngoc/ml/crnn/tts/InkData_line_processed/20140603_0043_BCCTC_tg_2_5.png'
    img = cv2.imread(path)
    attention = attention_ocr()
    res = attention.predict(img)
    img = Image.fromarray(img)
    plt.imshow(img)
    print(res)