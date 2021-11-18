import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset,DataLoader

class SeqInceptionBlock(nn.Module):
    def __init__(self,in_nc,out_nc=100):
        super(SeqInceptionBlock,self).__init__()
        # 1x1 convolution branch
        self.conv_1x1 = nn.Conv1d(in_nc, out_nc, kernel_size=1)
        # 3x3 convolution branch
        self.conv_3x3 = nn.Conv1d(in_nc, out_nc, kernel_size=3,padding=1)
        # 5x5 convolution branch
        self.conv_5x5 = nn.Conv1d(in_nc, out_nc, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_pool = nn.Conv1d(out_nc*3,out_nc,kernel_size=1)
    def forward(self,x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5], dim=1)
        x_out = self.conv_pool(x_out)
        return x_out
    
class AttentionBlock(nn.Module):
    def __init__(self,in_nc, num_input=3,base_nf=64):
        super(AttentionBlock,self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_nc, base_nf, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(base_nf,num_input),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        out = self.main(x)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self,in_nc=100,base_nf=64):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv1d(in_nc,base_nf,kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(base_nf,in_nc,kernel_size=3,padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
from torch.nn.utils.rnn import pad_sequence

class Arch(nn.Module):
    def __init__(self,in_nc,num_batch,out_nc=100,num_input=3,num_res=3,num_gru=2,base_nf=64):
        super(Arch,self).__init__()
        self.attn = AttentionBlock(in_nc,num_input,base_nf)
        self.encoders = nn.ModuleList()
        for i,_ in enumerate(range(num_input)):
            self.encoders.append(SeqInceptionBlock(in_nc-i,out_nc))
        model = []
        for _ in range(num_res):
            model.append(ResidualBlock(out_nc,base_nf))
        self.residual = nn.Sequential(*model)
        self.gru = nn.GRUCell(
            out_nc*2,num_gru)

    def forward(self,inputs,input2):
        raw = inputs[0]
        attn = self.attn(raw)
        outs = []
        for i,ins in enumerate(inputs):
            res = self.encoders[i](ins)
            outs.append(res)
        outs = torch.cat(outs,dim=2)
        outs = torch.einsum('bi,bji->bj',attn,outs).unsqueeze_(-1)  
        outs = self.residual(outs)
        final_input = torch.cat([input2,outs],dim=1).transpose(1,2)
        for ins in final_input:
            try:
                hx = self.gru(ins,hx)
            except UnboundLocalError:
                hx = self.gru(ins)
        logit = torch.sigmoid(hx)
        return logit
        
        
inputs = np.random.rand(20,7201)
inputs_grad_1 = np.diff(inputs)
inputs_grad_2 = np.diff(inputs,2)

inputs = torch.from_numpy(inputs).float().unsqueeze(-1)
inputs_grad_1 = torch.from_numpy(inputs_grad_1).float().unsqueeze(-1)
inputs_grad_2 = torch.from_numpy(inputs_grad_2).float().unsqueeze(-1)

data = [inputs,inputs_grad_1,inputs_grad_2]
pretrained = torch.randn(20,100,1)

model = Arch(7201,20)
out = model(data,pretrained) 
print(out)