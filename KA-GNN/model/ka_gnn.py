import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling




device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


'''



def pool_subgraphs_node(out, batched_graph):
    # 将整图的输出按照子图数量拆分成子图的输出
    subgraphs = dgl.unbatch(batched_graph)

    # 对每个子图进行池化操作，这里使用了平均池化
    pooled_outputs = []
    ini = 0
    for subgraph in subgraphs:
        # 获取子图节点的数量
        num_nodes = subgraph.num_nodes()
        
        # 根据子图的节点数量对整图的输出进行切片
        start_idx = ini
        end_idx = start_idx + num_nodes
        sg_out = out[start_idx:end_idx]
        ini += num_nodes
        # 计算每个子图的平均池化表示
        #print(sg_out.shape)
        #pooled_out = F.avg_pool2d(sg_out, kernel_size=num_nodes)  # 使用节点数作为池化核的大小
        pooled_out = F.adaptive_avg_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        pooled_outputs.append(pooled_out)

    return torch.stack(pooled_outputs)


def pool_subgraphs_edge(out, batched_graph):
    # 将整图的输出按照子图数量拆分成子图的输出
    subgraphs = dgl.unbatch(batched_graph)

    # 对每个子图进行池化操作，这里使用了平均池化
    pooled_outputs = []
    ini = 0
    for subgraph in subgraphs:
        # 获取子图节点的数量
        num_edges = subgraph.num_edges()
        
        # 根据子图的节点数量对整图的输出进行切片
        start_idx = ini
        end_idx = start_idx + num_edges
        sg_out = out[start_idx:end_idx]
        ini += num_edges
        # 计算每个子图的平均池化表示
        #print(sg_out.shape)
        #pooled_out = F.avg_pool2d(sg_out, kernel_size=num_nodes)  # 使用节点数作为池化核的大小
        pooled_out = F.adaptive_avg_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()
        #pooled_out = F.adaptive_max_pool1d(sg_out.unsqueeze(0).permute(0, 2, 1), 1).squeeze()

        pooled_outputs.append(pooled_out)

    return torch.stack(pooled_outputs)

'''

class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True):
        super(KAN_linear,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        #This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        
        # #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        # y =  torch.sum(c * self.fouriercoeffs[0:1], (-2, -1)) 
        # y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        # if self.addbias:
        #     y += self.bias
        # #End fuse
        
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        y = y.view(outshape)
        return y
    



class NaiveFourierKANLayer(nn.Module):
    def __init__(self, in_feats, out_feats, gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.in_feats = in_feats
        self.out_feats = out_feats

        # Fourier coefficients as parameters
        self.fouriercoeffs = nn.Parameter(torch.randn(2, out_feats, in_feats, gridsize) / 
                                          (np.sqrt(in_feats) * np.sqrt(gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(out_feats))


    def forward(self, g, x):
        # Assuming x is the input feature matrix of shape (N, in_feats)
        with g.local_scope():
            g.ndata['h'] = x
            
            g.update_all(message_func=self.fourier_transform, reduce_func=fn.sum(msg='m', out='h'))
            # If there is a bias, add it after message passing
            if self.addbias:
                g.ndata['h'] += self.bias

            return g.ndata['h']

    def fourier_transform(self, edges):
        # Access the source node feature
        src_feat = edges.src['h']  # Shape: (E, in_feats)

        # Prepare Fourier basis functions
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=src_feat.device), (1, 1, 1, self.gridsize))
        src_rshp = src_feat.view(src_feat.shape[0], 1, src_feat.shape[1], 1)
        cos_kx = torch.cos(k * src_rshp)
        sin_kx = torch.sin(k * src_rshp)
        
        # Reshape for multiplication
        cos_kx = torch.reshape(cos_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))
        sin_kx = torch.reshape(sin_kx, (1, src_feat.shape[0], src_feat.shape[1], self.gridsize))

        # Perform Fourier transform using einsum
        m = torch.einsum("dbik,djik->bj", torch.concat([cos_kx, sin_kx], axis=0), self.fouriercoeffs)

        # Returning the message to be reduced
        return {'m': m}




class KA_GNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)

        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))

        #self.layers.append()
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
        

        #self.layers.append(KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias))
        #self.layers.append(NaiveFourierKANLayer(hidden_feat, out_feat, grid_feat, addbias=use_bias))


        #self.layers.append(NaiveFourierKANLayer(out_feat, out_feat, grid_feat, addbias=use_bias))
        self.linear_1 = KAN_linear(hidden_feat, out, 1, addbias=True)
        #self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=True)
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()

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

        #y = pool_subgraphs_node(h, g)
        out = self.Readout(y)    
        return out
    
    def get_grad_norm_weights(self) -> nn.Module:
        # 返回需要进行梯度范数计算的参数
        return self.parameters()
    


class KA_GNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(KA_GNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.kan_line = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        # 初始化隐藏层的 KAN 层
        for _ in range(num_layers - 1):
            self.layers.append(NaiveFourierKANLayer(hidden_feat, hidden_feat, grid_feat, addbias=use_bias))
       
        self.linear_1 = KAN_linear(hidden_feat, out_feat, grid_feat, addbias=use_bias)
        self.linear_2 = KAN_linear(out_feat, out, grid_feat, addbias=use_bias)
        

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
        h = self.kan_line(features)
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                h = layer(g, h)  # 假设 KANLayer 被改造为接受 (graph, features)
                
            else:
                h = layer(h)  # 最后一层（线性）
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
    

if __name__ == '__main__':
    # 创建一个简单的图
    num_nodes = 10  # 节点数
    num_edges = 30  # 边数
    g = dgl.rand_graph(num_nodes, num_edges)  # 随机生成一个有向图

    # 为节点分配特征
    node_features = torch.randn(num_nodes, 5)  # 假设输入特征维度为5

    # 初始化模型
    model = KanGNN(in_feat=5, hidden_feat=10, out_feat=3, grid_feat=100, num_layers=3, use_bias=True)

    # 前向传播
    logits = model(g, node_features)
    
    # 打印输出
    print("Logits:", logits)
    