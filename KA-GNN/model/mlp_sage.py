import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from dgl.nn import SAGEConv
from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling



device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

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




class MLPGNN_two(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(MLPGNN_two, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.layers = nn.ModuleList()

        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.kan_line = nn.Linear(in_feat, hidden_feat, bias=use_bias)

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
        self.linear_1 = nn.Linear(hidden_feat, out,bias=True)
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
    


class MLPGNN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, out, grid_feat, num_layers, pooling, use_bias=False):
        super(MLPGNN, self).__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        #self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.lin_in = nn.Linear(in_feat, hidden_feat,bias=use_bias)
        self.layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        # 初始化隐藏层的 KAN 层
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_feat, hidden_feat, 'mean'))
        # 输出层
        #self.layers.append(nn.Linear(hidden_feat, out_feat, bias=use_bias))
        self.linear_1 = nn.Linear(hidden_feat, out_feat,bias=use_bias)
        self.linear_2 = nn.Linear(out_feat, out, bias=use_bias)
        
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
    

if __name__ == '__main__':
    # 创建一个简单的图
    num_nodes = 10  # 节点数
    num_edges = 30  # 边数
    g = dgl.rand_graph(num_nodes, num_edges)  # 随机生成一个有向图

    # 为节点分配特征
    node_features = torch.randn(num_nodes, 5)  # 假设输入特征维度为5

    # 初始化模型
    model = MLPGNN(in_feat=5, hidden_feat=10, out_feat=3, grid_feat=100, num_layers=3, use_bias=True)

    # 前向传播
    logits = model(g, node_features)
    
    # 打印输出
    print("Logits:", logits)
    