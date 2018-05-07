import torch
import torch.nn as nn
from torch.autograd import Variable

class BatchPC(nn.Module):
    def __init__(self, din, dout, momentum=0.1):
        super(BatchPC, self).__init__()

        self.register_buffer('pivot', torch.linspace(0, 1, din).view(din, 1))
        self.register_buffer('rC', torch.eye(din))
        self.register_buffer('Q', torch.eye(dout, din))
        self.momentum = momentum
        self.din = din
        self.dout = dout
    
    def forward(self, input):
        if self.training:
            n = input.size(0)
            C = torch.matmul(input.data.permute(1, 0), input.data)/n
            #print(type(C))
            #print(self.rC)
            dC = C - self.rC
            self.rC.add_(self.momentum * dC)
            
            es, ev = torch.eig(self.rC, eigenvectors=True)
            # top k eigenvalues
            es = es[:self.dout, 0:1]                 # dout, 1
            ev = ev[:, :self.dout].permute(1, 0)   # dout, din
            # align with pivot vector
            pos = torch.sign(ev.matmul(self.pivot))  # dout, 1
            ev = pos * ev
            # compute precond matrix
            self.Q = (1.0/es.sqrt()) * ev   # dout, din
            # print(self.Q.size())
            # print(es.size())
            # print(ev.size())
            # print(input.size())
        return torch.nn.functional.linear(input, Variable(self.Q, requires_grad=False))