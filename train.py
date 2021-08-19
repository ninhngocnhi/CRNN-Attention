# coding:utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import src.utils as utils
import src.dataset as dataset
import time
from src.utils import alphabet
from src.utils import weights_init

import models.crnn_attention as crnn
print(crnn.__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--path',  default='data/', help='where to ascess image')
parser.add_argument('--trainlist',  default='data/train')
parser.add_argument('--vallist',  default='data/test')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=21, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--encoder', type=str, default='expr/encoder_450.pth', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='expr/decoder_450.pth', help='path to decoder (to continue training)')
parser.add_argument('--experiment', default='./expr/attentioncnn', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=71, help='the width of the featuremap out from cnn')
opt = parser.parse_args()
print(opt)

SOS_token = 0
EOS_TOKEN = 1              
BLANK = 2                  

device = torch.device("cuda:0" if opt.cuda else "cpu")

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir -p {0}'.format(opt.experiment))       

opt.manualSeed = random.randint(1, 10000)  
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = None
train_dataset = dataset.listDataset(path = opt.path, list_file =opt.trainlist, transform=transform)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
    
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.listDataset(path = opt.path, list_file =opt.vallist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
test_train = dataset.listDataset(path = opt.path, list_file =opt.trainlist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
nclass = len(alphabet) + 3         
nc = 3
converter = utils.strLabelConverterForAttention(alphabet)
criterion = torch.nn.NLLLoss()             


encoder = crnn.CNN(opt.imgH, nc, opt.nh)
decoder = crnn.AttentionDec(nclass, opt.nh)       
encoder.apply(weights_init)
decoder.apply(weights_init)

# state_dict_init = encoder.state_dict()
# state_dict = torch.load("/content/expr/attentioncnn/netCRNN_50.pth")
# pretrained_dict = {k: v for k, v in state_dict.items() if "cnn" in k }
# state_dict_init.update(pretrained_dict)
# encoder.load_state_dict(state_dict_init)

if opt.encoder:
    print('loading pretrained encoder model from %s' % opt.encoder)
    encoder.load_state_dict(torch.load(opt.encoder))
if opt.decoder:
    print('loading pretrained encoder model from %s' % opt.decoder)
    decoder.load_state_dict(torch.load(opt.decoder))
print(encoder)
print(decoder)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr)
else:
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)


def val(encoder, decoder, criterion, batchsize, dataset, teach_forcing=False, max_iter=100, num = 10):
    print('Start')

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False
    encoder.eval()
    decoder.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batchsize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        utils.loadData(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1                    

        decoded_words = []
        decoded_label = []
        decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)
        encoder_outputs = encoder(image)           
        target_variable = target_variable.to(device)
        decoder_input = target_variable[0].to(device)
        decoder_hidden = decoder.initHidden(b).to(device)
        loss = 0.0
        if not teach_forcing:
            for di in range(1, target_variable.shape[0]):  
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])  
                loss_avg.add(loss)
                decoder_attentions[di-1] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze(1)    
                decoder_input = ni
                if ni == EOS_TOKEN:
                    decoded_words.append('<EOS>')
                    decoded_label.append(EOS_TOKEN)
                    break
                else:
                    decoded_words.append(converter.decode(ni))
                    decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1
        if i%num == 0:
            texts = cpu_texts[0]
            print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

    accuracy = n_correct / float(n_total)
    print('Loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, teach_forcing_prob=1):
    '''
        target_label:采用后处理的方式，进行编码和对齐，以便进行batch训练
    '''
    data = train_iter.next()
    cpu_images, cpu_texts = data
    b = cpu_images.size(0)
    target_variable = converter.encode(cpu_texts)
    utils.loadData(image, cpu_images)
    encoder_outputs = encoder(image)              
    target_variable = target_variable.to(device)
    decoder_input = target_variable[0].to(device)     
    decoder_hidden = decoder.initHidden(b).to(device)
    loss = 0.0
    teach_forcing = True if random.random() > teach_forcing_prob else False
    if teach_forcing:
        for di in range(1, target_variable.shape[0]):         
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            loss += criterion(decoder_output, target_variable[di])         
            decoder_input = target_variable[di] 
    else:
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])  
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni
    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


if __name__ == '__main__':
    t0 = time.time()
    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader)-1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            cost = trainBatch(encoder, decoder, criterion, encoder_optimizer, 
                              decoder_optimizer, teach_forcing_prob=opt.teaching_forcing_prob)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opt.niter, i, len(train_loader), loss_avg.val()), end=' ')
                loss_avg.reset()
                t1 = time.time()
                print('time elapsed %d' % (t1-t0))
                t0 = time.time()

        # do checkpointing
        if epoch % opt.saveInterval == 0:
            print("For train\n")
            val(encoder, decoder, criterion, 1, dataset=test_train, teach_forcing=False, num = 20)
            print("For test\n")
            val(encoder, decoder, criterion, 1, dataset=test_dataset, teach_forcing=False, num = 20)            # batchsize:1
            torch.save(
                encoder.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch))
            torch.save(
                decoder.state_dict(), '{0}/decoder_{1}.pth'.format(opt.experiment, epoch))
