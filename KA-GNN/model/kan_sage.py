import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from dgl.nn import SAGEConv
from kan import KAN
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling




device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")





class KANGNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KANGNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()
        #KAN(width=[2,5,1], grid=5, k=3, seed=0)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = KAN(width=[in_feat,5,hidden_feat], grid=grid_feat, k=3, seed=0)
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_feat, hidden_feat, 'mean'))

        #self.layers.append()
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

        #self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))


        #self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        self.linear_1 = KAN(width=[hidden_feat,5,out], grid=grid_feat, k=3, seed=0)
        #self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=True)
        
        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

        
    def forward(self, g, h):
        h = self.kan_line(h)

        for i, layer in enumerate(self.layers):
            m = layer(g, h)  # 假设 KANLayer 被改造为接受 (graph, features)
            #h = self.leaky_relu(torch.add(m,h)
            h = nn.functional.leaky_relu(torch.add(m, h))
        
        if self.pooling == 'avg':
            y = self.avgpool(g, h)
            #y1 = pool_subgraphs_node(out_1, g_graph)
            #y2 = pool_subgraphs_node(out_2, lg_graph)
            #y3 = pool_subgraphs_node(out_3, fg_graph)


        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)    
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        # 返回需要进行梯度范数计算的参数
        return self.parameters()
    


class KANGNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KANGNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        #self.lin_in = nn.Linear(in_feat, hidden_feat,bias=use_bias)
        self.lin_in = KAN(width=[in_feat,5,hidden_feat], grid=grid_feat, k=3, seed=0)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        # 初始化隐藏层的 KAN 层
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_feat, hidden_feat, 'mean'))
        # 输出层
        #self.layers.append(nn.Linear(hidden_feat, out_feat, bias=use_bias))
        self.linear_1 = KAN(width=[hidden_feat,5,out_feat], grid=grid_feat, k=3, seed=0)
        self.linear_2 = KAN(width=[out_feat,5,out], grid=grid_feat, k=3, seed=0)
        
        #self.linear = nn.Linear(hidden_feat, out, bias=use_bias)
        
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()
        
        layers_kan = [
                        #nn.Linear(self.hidden_size*2, self.hidden_size),
                        self.linear_1,
                        #nn.Sigmoid(),
                        self.leaky_relu,
                        self.linear_2,
                        nn.Sigmoid()
                        ]
        
        self.Readout = nn.Sequential(*layers_kan)  

    def forward(self, g, features):
        h = self.lin_in(features)
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                h = layer(g, h)  # 假设 KANLayer 被改造为接受 (graph, features)
                
            else:
                h = layer(h)  # 最后一层（线性）
        if self.pooling == 'avg':
            y = self.avgpool(g, h)
            

        elif self.pooling == 'max':
            y = self.maxpool(g, h)
            
        
        elif self.pooling == 'sum':
            y = self.sumpool(g, h)


        else:
            print('No pooling found!!!!')

        out = self.Readout(y)     
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        # 返回需要进行梯度范数计算的参数
        return self.parameters()
    