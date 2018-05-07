# preconditioning on stochastic optimization
# learn the precondition matrix, regularize the correlation matrix to be identity

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision
# import torchvision.models as models
import torch.nn.functional as F
# import torch.optim as optim
# import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as DD

from glob import glob
from scipy import misc
from PIL import Image
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import shutil
import time
import numpy as np
import random
import scipy
import scipy.io as sio
from collections import defaultdict
import sys


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='PyTorch batch preconditioning on 1 fc layer')
# parser.add_argument('-d', '--data_path', default='/nfs/bigbrain/boyu/Projects/Scattering/data/saxs_gisaxs', type=str, metavar='DP',
#                     help='data_path')
parser.add_argument('--k', default=300, type=int, metavar='K',
                    help='number of decompose dimension (default: 30), if -1, use full input dimension')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=1000, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight_decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--evaluate', type=str2bool, default=False, metavar='EV',
                    help='evaluate model on validation set')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--gpuid', type=int, default=0, metavar='G',
                    help='gpuid')
parser.add_argument('--precondition', type=str2bool, default=True, metavar='P',
                    help='whether to use precondition')

args = parser.parse_args()
print(args)
if not args.precondition:
    print('only work for precondition')
    sys.exit()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
use_cuda = torch.cuda.is_available()
# print(use_cuda)
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed_all(args.seed)

cudnn.benchmark = True


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)


def plot_fig(vector, save_path, save_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, vector.shape[0]), vector)
    plt.savefig(os.path.join(save_path, save_name))
    plt.close(fig)


class Network(nn.Module):
    def __init__(self, input_dim, k, output_dim=1):
        super(Network, self).__init__()
        self.precondition = nn.Sequential(nn.Linear(input_dim, k))
        self.main = nn.Sequential(
            nn.Linear(k, output_dim))
    def forward(self, input):
        pc_input = self.precondition(input)
        output = self.main(pc_input)
        return pc_input, output


class mnist_balanced(DD.Dataset):
    def __init__(self, train=True, root='data/mnist/', transform=torchvision.transforms.ToTensor(), labels=[4,7]):
        super(mnist_balanced, self).__init__()
        self.labels = labels
        mnist = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=True)
        if train:
            mnist_labels = mnist.train_labels
        else:
            mnist_labels = mnist.test_labels
        nClass = 2

        # get idxs[0], ..., idxs[9]
        lb2idxs = defaultdict(list)
        for i, v in enumerate(mnist_labels):
            lb2idxs[v].append(i)
        idxs = []
        for key in labels:
            idxs.append(lb2idxs[key])
        del lb2idxs

        self.mnist = mnist
        self.nClass = nClass
        self.idxs = idxs
        self.shuffle()

    def shuffle(self):
        for key in range(self.nClass):
            random.shuffle(self.idxs[key])

    def __getitem__(self, index):
        idx, cls = divmod(index, self.nClass)
        idx = self.idxs[ cls ][ idx % len(self.idxs[cls]) ]
        new_target = -1 if self.mnist[idx][1] == self.labels[0] else 1
        return self.mnist[idx][0].view([-1]), new_target

    def __len__(self):
        tosum = [len(self.idxs[key]) for key in range(self.nClass)]
        return sum(tosum)


def run_epoch(dataLoder, model, criterion, num_imgs, is_train=False, optimizer=None):
    if is_train:
        model.train()
    else:
        model.eval()
    losses = 0
    regularizer_losses = 0
    I = Variable(torch.eye(args.k)).cuda()
    regularizer_criterion = nn.MSELoss().cuda()
    # num_batches = len(dataLoder)
    # print(num_batches)
    weight = 20
    for i, (input_img, input_label) in enumerate(dataLoder, 0):
        batch_size = input_img.size(0)
        input_img = input_img #/ float(l2dis)
        input_img = input_img.cuda()
        # input_img = torch.matmul(input_img, P_half_inv)
        input_img_var = Variable(input_img)
        input_label_var = Variable(input_label).float().cuda()
        pc_input, output_label = model(input_img_var)
        loss = criterion(output_label, input_label_var) * batch_size / float(num_imgs)
        losses += loss.data[0]

        C = torch.matmul(pc_input.permute(1,0), pc_input) / batch_size
        regularizer_loss = regularizer_criterion(C, I)
        regularizer_losses += regularizer_loss.data[0]

        all_loss = loss + weight * regularizer_loss

        if is_train:
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

    return losses, regularizer_losses


def main():
    trSet = mnist_balanced(train=True, root='data/mnist/', labels=[4,7])
    tstSet = mnist_balanced(train=False, root='data/mnist/', labels=[4,7])
    trLD = DD.DataLoader(trSet, batch_size=args.batch_size, sampler=DD.sampler.RandomSampler(trSet), num_workers=args.workers, pin_memory=True)
    tstLD = DD.DataLoader(tstSet, batch_size=args.batch_size, sampler=DD.sampler.SequentialSampler(tstSet), num_workers=args.workers, pin_memory=True)

    print('training samples: %d' %(len(trSet)))
    print('testing samples: %d' %(len(tstSet)))

    data_dim = 784
    if args.k == -1:
        args.k = data_dim
    
    model = Network(data_dim, args.k, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0)
    criterion = nn.MSELoss().cuda()

    t = time.time()
    save_name_prefix = 'minibatch_regularize_c_pre-%s_k-%d_decay-%.5f_lr-%.4f_batchsize-%d'%(args.precondition, args.k, args.weight_decay, args.lr, args.batch_size)
    logger = Logger(os.path.join('result', '%s_log.txt'%(save_name_prefix)))

    loss_tr_all = []
    loss_tst_all = []
    reg_loss_tr_all = []
    reg_loss_tst_all = []
    for epoch in range(args.epochs):
        loss_tr, reg_loss_tr = run_epoch(trLD, model, criterion, len(trSet), is_train=True, optimizer=optimizer)
        loss_tst, reg_loss_tst = run_epoch(tstLD, model, criterion, len(tstSet), is_train=False, optimizer=None)
        loss_tr_all.append(loss_tr)
        loss_tst_all.append(loss_tst)
        reg_loss_tr_all.append(reg_loss_tr)
        reg_loss_tst_all.append(reg_loss_tst)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain loss: %.5f, %.5f' % (loss_tr, reg_loss_tr))
        logger.write('\ttest loss: %.5f, %.5f ' % (loss_tst, reg_loss_tst))

        plot_fig(np.array(loss_tr_all), 'result', '%s_tr.png'%(save_name_prefix))
        plot_fig(np.array(loss_tst_all), 'result', '%s_tst.png'%(save_name_prefix))
        to_save = dict()
        to_save['tr'] = np.array(loss_tr_all)
        to_save['tst'] = np.array(loss_tst_all)
        sio.savemat(os.path.join('result', '%s_loss.mat'%(save_name_prefix)), to_save)


if __name__ == '__main__':
    main()
