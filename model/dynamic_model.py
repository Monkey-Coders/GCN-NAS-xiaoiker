import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def get_chebshev_approximation(A, indices):
    soft = nn.Softmax(-2)
    A_ch4s = soft((8*torch.pow(A, 4)- 8*torch.pow(A, 2)+torch.eye(A.size(-1)))/A.size(-1))
    A_ch4 = (8*torch.pow(A, 4)- 8*torch.pow(A, 2) +torch.eye(A.size(-1)))
    A_ch3 = (4*torch.pow(A, 3)-3*A)
    A_ch2 = (2*torch.pow(A, 2) - torch.eye(A.size(-1)))
    approximations = [A, A_ch4s, A_ch4, A_ch3, A_ch2]
    return torch.stack([approximations[i] for i in indices], dim=0)


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


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

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
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


        
class unit_gtcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gtcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))# I think this the Bk in the paper.
        nn.init.constant(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset # How many layers in each sub-Network. 

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        self.conv_T1 = nn.ModuleList()
        self.conv_T2 = nn.ModuleList()
        
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
            
            self.conv_T1.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))# To build graph from temporal infomation.
            self.conv_T2.append(nn.Conv2d(in_channels, inter_channels, (9,1), padding=(4, 0)))

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

        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x): 
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA # Is this A the adjecent Matrix? PA is Bk?

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
            
            A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
            A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
            
            A1 = A[i] + A1 + A_T1 # Means Ak+Bk+Ck+Tk in Eq(3), in line 95: A = A + B
            
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class unit_gtcn_C(nn.Module):
    def __init__(self, in_channels, out_channels, A,weights, coff_embedding=4, num_subset=3):
        super(unit_gtcn_C, self).__init__()
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
        # Define if we need spatial depending on the weight
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
        #self.conv_ST21 = nn.ModuleList()
        #self.conv_ST22 = nn.ModuleList()
        
        for i in range(self.num_subset):
            if self.use_spatial:
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))# There are 3 sub-Networks in the Unit, Here means all the Kernel_size=1
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
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
        
       
        approximation_indices= [i for i, x in enumerate(weights[0:5]) if x > 0.1]
        self.approximations = get_chebshev_approximation(self.A, approximation_indices)

        
        for m in self.modules():# return all the modules in the model
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, weights):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        
        #Note not include the PA during searching
        approximations_cuda = [approximation.cuda(x.get_device()) for approximation in self.approximations]
        # TODO
        # Verify that this works
        A = sum([weight * approximation for weight, approximation in zip(weights, approximations_cuda)])
        y = None

        for i in range(self.num_subset):
            operations = []
            if self.use_spatial:
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V) #: Conv out: N, C, T, V -->  N , C*T, V
                A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V, and / A1.size(-1) means normalize?? # Note: A1 here is Ck in the Eq. 
                operations.append(A1)
            if self.use_temporal:
                A_T1= self.conv_T1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
                A_T2 = self.conv_T2[i](x).view(N, self.inter_c * T, V) 
                A_T1 = self.soft(torch.matmul(A_T1, A_T2) / A_T1.size(-1))
                operations.append(A_T1)
            if self.use_spatial_temporal:
                A_ST11= self.conv_ST11[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
                A_ST12 = self.conv_ST12[i](x).view(N, self.inter_c * T, V) 
                A_ST11 = self.soft(torch.matmul(A_ST11, A_ST12) / A_ST11.size(-1))
                operations.append(A_ST11)
            #A_ST21= self.conv_ST21[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)#: Conv out:N, C, T, V --> N , V, C, T --> N , V, C*T
            #A_ST22 = self.conv_ST22[i](x).view(N, self.inter_c * T, V) 
            #A_ST21 = self.soft(torch.matmul(A_ST21, A_ST22) / A_ST21.size(-1))
            
            #A1 = A[i] + weights[5]*A1 + weights[6]*A_T1 + weights[7]*A_ST11 #+ weights[8]*A_ST21 # Means Ak+Bk+Ck+Tk in Eq(3), in line 95: A = A + B
            A1 = A + sum([weight * operation for weight, operation in zip(weights[5:], operations)])
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))# Means f_out in Eq(3)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)         

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, weights, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        #self.gcn1 = unit_gcn(in_channels, out_channels, A)
        # Need to find out what gcn1 is and what tcn1 is
        self.gcn1 = unit_gtcn_C(in_channels, out_channels, A, weights)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        #self.tcn1 = unit_tcn_G(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.weights = weights
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, weight):
        x = self.tcn1(self.gcn1(x, weight)) + self.residual(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, weights = None, layers=None, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, criterion=nn.CrossEntropyLoss()):
        super(Model, self).__init__()

        if graph is None or weights is None or layers is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        #self.register_buffer('Weights_A', torch.ones(10,8)*0.125)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        # Get the size of the weights
        layers_size, function_modules_size = weights.shape
        # We need to create an array of layers based on the layer attribute 
        self.layers = []
        self.l1 = TCN_GCN_unit(in_channels=3, out_channels=64, A=A, weights=weights[0], stride=1, residual=False)
        for i , layer in enumerate(layers):
            setattr(self,"l"+str(i+1), TCN_GCN_unit(in_channels=layer["in_channels"], out_channels=layer["out_channels"], A=A, weights=layer["weights"], stride=layer["stride"], residual=layer["residual"]))
            self.layers.append(getattr(self,"l"+str(i+1)))
        

        self.fc = nn.Linear(256, num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        
        self._criterion = criterion
        
  
    def forward(self, x, weights):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

     
        
        for i, layer in enumerate(self.layers):
            x = layer(x, weights[i])

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)

#    def _loss(self, x, target):
#        logits = self(x)
#        return self._criterion(logits, target)
#        
#    def arch_parameters(self):
#        return self.archi_params # should be a iterable one

