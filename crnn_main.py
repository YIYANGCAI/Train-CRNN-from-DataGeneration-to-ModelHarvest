#!/usr/bin/python
#!-*-coding:utf-8-*-

"""
load the pre-training model train
nclass is the same as pretrained model
"""
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset
import models.crnn as crnn
from binascii import hexlify
from codecs import encode
import ast

#alphabet = '01234abcde'
alphabet = '01234abcde一二三四五'

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', default='./datasets/second_data/train_lmdb/', help='path to dataset')
parser.add_argument('--valroot', default='./datasets/second_data/test_lmdb/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="")
parser.add_argument('--alphabet', type=str, default=alphabet)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=5, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', default=True,
                    help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
#print(opt)

# find the alphabet of icdar2015-4.3 demo


if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir ./{0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True  # improve speed,no spending

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1
print(nclass-1)
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    crnn.load_state_dict(torch.load(opt.crnn))
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

opt.adam = True
# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    print("adam")
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    # optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    print("max_iter", max_iter, "len(data_loader)", len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        # print(data)
        i += 1
        cpu_images, cpu_texts = data

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)

        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        list_cpu_texts = []
        for i in cpu_texts:
            list_cpu_texts.append((i).encode('utf-8').decode('utf-8', 'strict'))

        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        if (i == 1):
            print(sim_preds)
            print(cpu_texts)
        #        cpu_texts = byte_to_zh(cpu_texts)
        # print("sim_preds",sim_preds)
        for pred, target in zip(sim_preds, list_cpu_texts):
            if (pred == target.lower()) | (pred == target):
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]

    for raw_pred, pred, gt in zip(raw_preds, sim_preds, list_cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    # print(image)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

if __name__ == '__main__':
    numLoss = 0
    lasttestLoss = 10000
    testLoss = 10000
    num = 0
    for epoch in range(1, opt.niter):
        print("epoch\t", epoch, "opt.niter\t", opt.niter)
        train_iter = iter(train_loader)
        #print(len(train_iter))
        i = 0
        while i < len(train_loader):
            print("i:",i)
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = trainBatch(crnn, criterion, optimizer)
            # print(cost)
            loss_avg.add(cost)
            print(loss_avg.val())
            i += 1
            # print(i,op# t.saveInterval,"Loss:",loss_avg.val())
        if epoch % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] '%(epoch, opt.niter, i, len(train_loader)))
            loss_avg.reset()

        if epoch % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if epoch % opt.saveInterval == 0:
            torch.save(crnn.state_dict(), './{}/CRNN_{}_{}.pth'.format(opt.experiment, epoch, i))
            #torch.save(crnn.state_dict(), '{0}/model.pth'.format(opt.experiment))
