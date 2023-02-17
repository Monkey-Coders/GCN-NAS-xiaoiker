import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def no_op(self, x):
    return x

def get_chebyshev_approximation(A, indices):
    soft = nn.Softmax(-2)
    A_ch4s = soft((8*torch.pow(A, 4)- 8*torch.pow(A, 2)+torch.eye(A.size(-1)))/A.size(-1))
    A_ch4 = (8*torch.pow(A, 4)- 8*torch.pow(A, 2) +torch.eye(A.size(-1)))
    A_ch3 = (4*torch.pow(A, 3)-3*A)
    A_ch2 = (2*torch.pow(A, 2) - torch.eye(A.size(-1)))
    approximations = [A, A_ch4s, A_ch4, A_ch3, A_ch2]
    return torch.stack([approximations[i] for i in indices], dim=0) if len(indices) >= 1 else []


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gtcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, weights, coff_embedding=4, num_subset=3):
        super(unit_gtcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 
        self.weights = weights

        self.use_spatial = weights[5] > 0.1
        self.use_temporal = weights[6] > 0.1
        self.use_spatial_temporal = weights[7] > 0.1

        if self.use_spatial:
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        
        if self.use_temporal:
            self.conv_T1 = nn.ModuleList()
            self.conv_T2 = nn.ModuleList()

        if self.use_spatial_temporal:
            self.conv_ST11 = nn.ModuleList()
            self.conv_ST12 = nn.ModuleList()
        
        for i in range(num_subset):
            if self.use_spatial:
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            
            #This is used even if we are not using spatial for some reason
            # Will have nothing to say because we are checking if use_spatial is true later
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            if self.use_temporal:
                self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
                self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
            
            if self.use_spatial_temporal:
                self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, 1))# To build graph from temporal infomation.
                self.conv_ST11.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))
                self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_ST12.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        
        approximation_indices= [i for i, x in enumerate(self.weights[0:5]) if x > 0.1]
        approximations = get_chebyshev_approximation(self.A, approximation_indices)
        
        approximations = [approximation.cuda(x.get_device()) for approximation in approximations]
        
        A = sum(approximations) if len(approximations) >= 1 else 0
        A = A + self.PA


        y = None

        for i in range(self.num_subset):
            # Note, if we are not using spatial we set the value to 0 
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) if self.use_spatial else 0
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) if self.use_spatial else 0
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  if self.use_spatial else 0

            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) if self.use_temporal else 0
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) if self.use_temporal else 0
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1)) if self.use_temporal else 0

            A_ST11= self.conv_ST11[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) if self.use_spatial_temporal else 0
            A_ST12 = self.conv_ST12[i](x).view(N, self.inter_c * T, V) if self.use_spatial_temporal else 0
            A_ST11 = self.soft(torch.matmul(A_ST11, A_ST12) / A_ST11.size(-1)) if self.use_spatial_temporal else 0

            A1 = A[i] + A1 + A_T1 + A_ST11

            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z
        
        y = self.bn(y)
        y+= self.down(x)
        return self.relu(y)

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, weights, residual=True, stride=1):
        super(TCN_GCN_unit, self).__init__()
        self.gtcn = unit_gtcn(in_channels, out_channels, A, weights)
        self.tcn = unit_tcn(out_channels, out_channels, stride=stride)
        
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
    def forward(self, x):
        x = self.tcn(self.gtcn(x)) + self.residual(x)
        return self.relu(x)

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, weights = None):
        super(Model, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.graph_input = graph
        self.graph_args = graph_args
        self.in_channels = in_channels
        self.weights = weights

        if graph is None or weights is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, weights=weights[0])
        #self.l2 = TCN_GCN_unit(64, 64, A, weights=weights[1])
        #self.l3 = TCN_GCN_unit(64, 64, A, weights=weights[2])
        #self.l4 = TCN_GCN_unit(64, 64, A, weights=weights[3])
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, weights=weights[1])
        #self.l6 = TCN_GCN_unit(128, 128, A, weights=weights[3])
        #self.l7 = TCN_GCN_unit(128, 128, A, weights=weights[6])
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, weights=weights[2])
        self.l9 = TCN_GCN_unit(256, 256, A, weights=weights[3])
        #self.l10 = TCN_GCN_unit(256, 256, A, weights=weights[9])
        #self.layers = [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9, self.l10]
        self.layers = [self.l1, self.l5, self.l8, self.l9]

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)


    def get_copy(self, bn=False):
        model_new = Model(num_class=self.num_class, num_point=self.num_point, num_person=self.num_person, graph=self.graph_input, graph_args=self.graph_args, in_channels=self.in_channels, weights=self.weights)
        if bn == False:
            for l in model_new.modules():
                if isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm1d):
                    l.forward = types.MethodType(no_op, l)
        model_new.load_state_dict(self.state_dict(), strict=False)
        return model_new
    
    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for layer in self.layers:
            x = layer(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)