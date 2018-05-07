# preconditioning on batch optimization (use whole dataset to update)
#

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
parser.add_argument('--k', default=30, type=int, metavar='K',
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
    def __init__(self, input_dim, output_dim=1):
        super(Network, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim))
    def forward(self, input):
        output = self.main(input)
        return output


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


# get the preconditioning matrix P\{-1/2}
# P = C + lamda* I
def getExactPreconditionMatrix(X, k, lambda_):
    [d, N] = X.shape

    if args.precondition:
        C = np.matmul(X, X.transpose())
        P = C + lambda_ * np.eye(d)
        es, ev = np.linalg.eig(P)
        es = np.real(es)
        ev = np.real(ev)
        idx = np.argsort(es)
        idx = idx[::-1]
        es = es[idx[:k]]
        ev = ev[:, idx[:k]]
        P_half_inv = np.matmul(np.matmul(ev, np.diag(1.0/np.sqrt(es))), ev.transpose())
    else:
        P_half_inv = np.eye(d)
    return P_half_inv


# get the preconditioning matrix P\{-1/2}
# P = C + lamda* I
def getPCAPreconditionMatrix(X, k, lambda_):
    [d, N] = X.shape

    if args.precondition:
        C = np.matmul(X, X.transpose())
        # P = C + lambda_ * np.eye(d)
        es, ev = np.linalg.eig(C)
        es = np.real(es)
        ev = np.real(ev)
        idx = np.argsort(es)
        idx = idx[::-1]
        es = es[idx[:k]]
        ev = ev[:, idx[:k]]
        # P_half_inv = np.matmul(np.matmul(ev, np.diag(1.0/np.sqrt(es))), ev.transpose())
        Q = np.matmul(ev, np.diag(1.0/np.sqrt(es)))
    else:
        Q = np.eye(d)
    return Q

# get the preconditioning matrix P\{-1/2}
# input is X: d*N
def getPreconditionMatrix(X, k, lambda_):
    [d, N] = X.shape
    # print(X.shape)
    X = scipy.sparse.csr_matrix(np.matrix(X))
    # U: d*k
    # S: k
    U, S, V = scipy.sparse.linalg.svds(X, k)
    # print(np.matmul(np.transpose(U),U))
    #sys.exit()
    # S = S[::-1]
    S_lambda = np.square(S) + lambda_
    # S_lambda = S_lambda / S_lambda[-1]
    # print(S_lambda)
    S_lambda = 1/np.sqrt(S_lambda)
    S_lambda_k = S_lambda[0]
    S_lambda = np.diag(S_lambda)
    T = np.matmul(np.matmul(U, np.diag(S)), V)
    print(np.linalg.norm(T-X))

    if args.precondition:
        P_half_inv = np.matmul(np.matmul(U, S_lambda), np.transpose(U)) + (np.eye(d) - np.matmul(U, np.transpose(U))) / S_lambda_k
    else:
        P_half_inv = np.eye(d)
    #print(np.matmul(np.matmul(U, S_lambda), np.transpose(U)))
    #print((np.eye(d) - np.matmul(U, np.transpose(U))) / S_lambda_k)
    # P_half_inv = np.eye(d)
    return P_half_inv
    # if k == -1: k = d
    # # use truncated SVD to decompose X = U * S * V'
    # if k < d:
    #     svd = sklearn.decomposition.TruncatedSVD(n_components=k)
    # else:
    #     svd = sklearn.decomposition.PCA(n_components=k)
    # svd.fit(X)
    #
    # U = svd.components_

def run_epoch(dataLoder, l2dis, P_half_inv, model, criterion, num_imgs, is_train=False, optimizer=None):
    if is_train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    losses = 0
    for i, (input_img, input_label) in enumerate(dataLoder, 0):
        batch_size = input_img.size(0)
        input_img = input_img / float(l2dis)
        input_img = input_img.cuda()
        input_img = torch.matmul(input_img, P_half_inv)
        input_img_var = Variable(input_img)
        input_label_var = Variable(input_label).float().cuda()
        output_label = model(input_img_var)
        loss = criterion(output_label, input_label_var) * batch_size / float(num_imgs)
        losses += loss.data[0]
        loss.backward()
    if is_train:
        optimizer.step()

    return losses


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
    if args.precondition:
        data_dim = args.k
    model = Network(data_dim, 1).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0)
    criterion = nn.MSELoss().cuda()

    # get precondition matrix for the whole training set
    trAllFea = []
    for batch_id, (input_img, input_label) in enumerate(trLD, 0):
        trAllFea.append(input_img)
    trAllFea = torch.cat(trAllFea, 0)
    trAllFea = trAllFea.permute(1,0)
    trAllFea = trAllFea.numpy() #/ len(trSet)

    # get condition number 
    Correlation = np.matmul(trAllFea, np.transpose(trAllFea)) / len(trSet)
    e_vals, e_vecs = np.linalg.eig(Correlation)
    # to_save = dict()
    # to_save['corre'] = Correlation
    # to_save['evals'] = e_vals
    # sio.savemat('corre_before.mat', to_save)
    Kappa = e_vals + args.weight_decay 
    Kappa = Kappa / Kappa[-1]
    print('before Kappa')
    print(np.sum(Kappa))
    print(np.mean(Kappa))
    # get new condition number
    # get avg l2 norm, preprocess data
    # l2dis = np.linalg.norm(trAllFea, axis=0)
    # l2dis = np.mean(l2dis)
    l2dis = 1
    print('mean norm')
    print(l2dis)
    trAllFea = trAllFea / l2dis / np.sqrt(len(trSet))
    # get precondion matrix
    ps_time = time.time()
    P_half_inv = getPCAPreconditionMatrix(trAllFea, args.k, args.weight_decay)
    pe_time = time.time()
    print('get preconditioning matrix time: %.4f'%(pe_time-ps_time))
    
    # get condition number 
    # loss_tr_alllFea = np.matmul(P_half_inv, trAllFea) * np.sqrt(len(trSet))
    trAllFea = np.matmul(P_half_inv.transpose(), trAllFea) * np.sqrt(len(trSet))
    Correlation = np.matmul(trAllFea, np.transpose(trAllFea)) / len(trSet)
    e_vals, e_vecs = np.linalg.eig(Correlation)
    # to_save = dict()
    # to_save['corre'] = Correlation
    # to_save['evals'] = e_vals
    # sio.savemat('corre_after.mat', to_save)
    # e_vals = e_vals[np.isreal(e_vals)]
    # print(e_vals)

    Kappa = e_vals + args.weight_decay
    Kappa = Kappa / Kappa[-1]
    print('after Kappa')
    print(np.sum(Kappa))
    print(np.mean(Kappa))

    # sys.exit()
    P_half_inv = torch.Tensor(P_half_inv).float().cuda()

    t = time.time()
    save_name_prefix = 'batch_1fc_pre-%s_k-%d_decay-%.5f_lr-%.4f'%(args.precondition, args.k, args.weight_decay, args.lr)
    logger = Logger(os.path.join('result', '%s_log.txt'%(save_name_prefix)))

    loss_tr_all = []
    loss_tst_all = []
    for epoch in range(args.epochs):
        loss_tr = run_epoch(trLD, l2dis, P_half_inv, model, criterion, len(trSet), is_train=True, optimizer=optimizer)
        loss_tst = run_epoch(tstLD, l2dis, P_half_inv, model, criterion, len(tstSet), is_train=False, optimizer=None)
        loss_tr_all.append(loss_tr)
        loss_tst_all.append(loss_tst)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain loss: %.5f' % (loss_tr))
        logger.write('\ttest loss: %.5f ' % (loss_tst))

        plot_fig(np.array(loss_tr_all), 'result', '%s_tr.png'%(save_name_prefix))
        plot_fig(np.array(loss_tst_all), 'result', '%s_tst.png'%(save_name_prefix))
        to_save = dict()
        to_save['tr'] = np.array(loss_tr_all)
        to_save['tst'] = np.array(loss_tst_all)
        sio.savemat(os.path.join('result', '%s_loss.mat'%(save_name_prefix)), to_save)


if __name__ == '__main__':
    main()
