#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:08:37 2024

@author: longlee
"""

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import random
import dgl
import statistics
import csv

from logzero import logger
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score,r2_score,mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_recall_curve, auc

from sklearn import metrics
from model.ka_gnn import KA_GNN,KA_GNN_two
from model.mlp_sage import MLPGNN,MLPGNN_two
from model.kan_sage import KANGNN, KANGNN_two
from torch.optim.lr_scheduler import StepLR
from ruamel.yaml import YAML
from utils.splitters import ScaffoldSplitter
from utils.graph_path import path_complex_mol
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class CustomDataset(Dataset):
    def __init__(self, label_list, graph_list):
        self.labels = label_list
        self.graphs = graph_list
        self.device = torch.device('cpu')  # 或 'cuda' 如果您使用 GPU

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 获取标签并移动到设备
        label = self.labels[index].to(self.device)
        
        # 获取单个图并移动到设备
        graph = self.graphs[index].to(self.device)
        
        return label, graph
    


def collate_fn(batch):
    labels, graphs = zip(*batch)  # 解压批次数据

    # 将标签堆叠成一个张量
    labels = torch.stack(labels)

    # 批次化图
    batched_graph = dgl.batch(graphs)

    return labels, batched_graph



def has_node_with_zero_in_degree(graph):
    if (graph.in_degrees() == 0).any():
                return True
    
    #for graph in graph_list:
    #    if (graph.in_degrees() == 0).any():
    #        return True
        
    return False


def has_isolated_hydrogens(samiles):
    # 获取分子中的原子
    molecule = Chem.MolFromSmiles(samiles)
    mol = Chem.AddHs(molecule)  # 加氢
    if molecule is None:
        return True
    
    atoms = mol.GetAtoms()
    if len(atoms) <= 2:
        return True
    
    # 遍历原子
    for atom in atoms:
        # 如果原子是氢原子且没有邻居
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            return True  # 存在孤立的氢原子
    
    return False  # 不存在孤立的氢原子





def conformers_is_zero(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 加氢
    AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42) 
    # 检查是否有构象
    num_conformers = mol.GetNumConformers()

    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # 检查图是否为连通图
    if G.number_of_nodes() > 0:
        is_connected = nx.is_connected(G)
        if num_conformers > 0 and is_connected == True:
            return True
    else:
        return False
    

    
def min_max_normalize(data):
    # 找到最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 对每个数据点应用Min-Max归一化公式
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

    return normalized_data, min_val, max_val


def inverse_min_max_normalize(x, min_val, max_val):
    # 对每个归一化后的数据点应用逆操作
    original_data = x * (max_val - min_val)
    return original_data

def is_file_in_directory(directory, target_file):
    file_path = os.path.join(directory, target_file)
    return os.path.isfile(file_path)


def unique(class_target):
    # 假设 y_true_np 是你的 NumPy 数组
    unique_classes, counts = np.unique(class_target, return_counts=True)

    # 打印唯一的类别和它们的出现次数
    for class_label, count in zip(unique_classes, counts):
        print(f"Class {class_label}: {count} samples")

    # 检查类别数量
    num_classes = len(unique_classes)
    if num_classes == 2:
        print("y_true_np 包含两个不同的类别.")
    else:
        print("y_true_np 不包含两个不同的类别.")

#others
def get_label():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['label']


#tox21,12     
def get_tox():
    """Get that default sider task names and return the side results for the drug"""
    
    return ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

#clintox,2
def get_clintox():
    
    return ['FDA_APPROVED', 'CT_TOX']

#sider,27
def get_sider():

    return ['Hepatobiliary disorders',
           'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
           'Investigations', 'Musculoskeletal and connective tissue disorders',
           'Gastrointestinal disorders', 'Social circumstances',
           'Immune system disorders', 'Reproductive system and breast disorders',
           'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
           'General disorders and administration site conditions',
           'Endocrine disorders', 'Surgical and medical procedures',
           'Vascular disorders', 'Blood and lymphatic system disorders',
           'Skin and subcutaneous tissue disorders',
           'Congenital, familial and genetic disorders',
           'Infections and infestations',
           'Respiratory, thoracic and mediastinal disorders',
           'Psychiatric disorders', 'Renal and urinary disorders',
           'Pregnancy, puerperium and perinatal conditions',
           'Ear and labyrinth disorders', 'Cardiac disorders',
           'Nervous system disorders',
           'Injury, poisoning and procedural complications']

#muv
def get_muv():
    
    return ['MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692',
            'MUV-712','MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846',
            'MUV-852',	'MUV-858','MUV-859']


def auc_function(y_true, y_pred):
    """
    计算两个张量之间的均方根误差（RMSE）。

    参数:
    - y_true (torch.Tensor): 真实标签的张量。
    - y_pred (torch.Tensor): 预测值的张量。

    返回:
    - torch.Tensor: RMSE 值。
    """
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    y_true = y_true.to(torch.float32)
    assert y_true.dtype == y_pred.dtype, "y_true and y_pred must have the same dtype"
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    unique(y_true_np)
    unique(y_pred_np)

    auc = roc_auc_score(y_true_np, y_pred_np)
    auc = auc.item()
    return auc



def creat_data(datafile, encoder_atom, encoder_bond,batch_size,train_ratio,vali_ratio,test_ratio):
    

    datasets = datafile

    directory_path = 'data/processed/'
    target_file_name = datafile +'.pth'

    if is_file_in_directory(directory_path, target_file_name):

        return True
    
    else:

        df = pd.read_csv('data/' + datasets + '.csv')#
        if datasets == 'tox21':
            smiles_list, labels = df['smiles'], df[get_tox()] 
            #labels = labels.replace(0, -1)
            labels = labels.fillna(0)

        if datasets == 'muv':
            smiles_list, labels = df['smiles'], df[get_muv()]  
            labels = labels.fillna(0)

        if datasets == 'sider':
            smiles_list, labels = df['smiles'], df[get_sider()]  

        if datasets == 'clintox':
            smiles_list, labels = df['smiles'], df[get_clintox()] 
    

        if datasets in ['hiv','bbbp','bace']:
            smiles_list, labels = df['smiles'], df[get_label()] 
            
        #labels = labels.replace(0, -1)
        #labels = labels.fillna(0)

        #smiles_list, labels = df['smiles'], df['label']        
        #labels = labels.replace(0, -1)
        
        #labels, min_val, max_val = min_max_normalize(labels)

        data_list = []
        feature_sets = ("atomic_number", "basic", "cfid", "cgcnn")
        for i in range(len(smiles_list)):
            if i % 10000 == 0:
                print(i)

            smiles = smiles_list[i]
            
            #if has_isolated_hydrogens(smiles) == False and conformers_is_zero(smiles) == True :

            Graph_list = path_complex_mol(smiles, encoder_atom, encoder_bond)
            if Graph_list == False:
                continue

            else:
                if has_node_with_zero_in_degree(Graph_list):
                    continue
                
                else:
                    data_list.append([smiles, torch.tensor(labels.iloc[i]),Graph_list])



        #data_list = [['occr',albel,[c_size, features, edge_indexs],[g,liearn_g]],[],...,[]]

        print('Graph list was done!')

        splitter = ScaffoldSplitter().split(data_list, frac_train=train_ratio, frac_valid=vali_ratio, frac_test=test_ratio)
        
        print('splitter was done!')
        

        
        train_label = []
        train_graph_list = []
        for tmp_train_graph in splitter[0]:
            
            train_label.append(tmp_train_graph[1])
            train_graph_list.append(tmp_train_graph[2])


        valid_label = []
        valid_graph_list = []
        for tmp_valid_graph in splitter[1]:
            valid_label.append(tmp_valid_graph[1])
            
            valid_graph_list.append(tmp_valid_graph[2])

        test_label = []
        test_graph_list = []
        for tmp_test_graph in splitter[2]:
            test_label.append(tmp_test_graph[1])
            test_graph_list.append(tmp_test_graph[2])

        #batch_size = 256

        torch.save({
            'train_label': train_label,
            'train_graph_list': train_graph_list,
            'valid_label': valid_label,
            'valid_graph_list': valid_graph_list,
            'test_label': test_label,
            'test_graph_list': test_graph_list,
            'batch_size': batch_size,
            'shuffle': True,  # 保存时假设你在创建 DataLoader 时使用了 shuffle=True
            # 其他必要信息
        }, 'data/processed/'+ datafile +'.pth')



def message_func(edges):
    """ 传递边的特征 """
    return {'feat': edges.data['feat']}

def reduce_func(nodes):
    """ 加和所有接收到的边特征 """
    num_edges = nodes.mailbox['feat'].size(1)  # 计算接收到的消息的数量
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  # 求平均
    #agg_feats = F.normalize(agg_feats, p=2, dim=1) 
    return {'agg_feats': agg_feats}

def update_node_features(g):
    """ 执行消息传递并更新节点特征 """
    g.send_and_recv(g.edges(), message_func, reduce_func)
    # 将加和后的边特征与原节点特征拼接
    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)

    return g




def add_noise(node_feat, noise=False):
    row, col = node_feat.shape  # 获取张量的形状
    
    if noise:
        # 如果需要添加噪声，生成噪声数据
        extension_tensor = torch.randn(row, 128 - col).to(device)
    else:
        # 否则添加全零的张量
        extension_tensor = torch.zeros(row, 128 - col).to(device)

    # 将两个张量沿着第二个维度（列）拼接
    extended_tensor = torch.cat([node_feat, extension_tensor], dim=1)

    return extended_tensor




def train(model, device, train_loader, valid_loader, optimizer, epoch):
    model.train()

    total_train_loss = 0.0
    train_num = 0

    
    for batch_idx, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        train_label_value = []
        y = data[0]
        train_label_value.append(torch.unsqueeze(y, dim=0))
        graph_list = update_node_features(data[1]).to(device)
        #node_features = graph_list.ndata['feat'].to(device)
        node_features = add_noise(graph_list.ndata['feat'],noise=True).to(device)
        #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
        output = model(graph_list, node_features).cpu()
        
        arr_label = torch.Tensor().cpu()
        arr_pred = torch.Tensor().cpu()
        for j in range(y.shape[1]):
            c_valid = np.ones_like(y[:, j], dtype=bool)
            c_label, c_pred = y[c_valid, j], output[c_valid, j]
            zero = torch.zeros_like(c_label)
            c_label = torch.where(c_label == -1, zero, c_label)
            
            arr_label = torch.cat((arr_label,c_label),0)
            arr_pred = torch.cat((arr_pred,c_pred),0)
        
        arr_pred = arr_pred.float()
        arr_label = arr_label.float()

        loss = loss_fn(arr_pred, arr_label)
        #loss = FocalLoss(arr_pred, arr_label)
        train_loss = torch.sum(loss)
        total_train_loss = total_train_loss + train_loss
        train_loss.backward()
        optimizer.step()

    '''if epoch+1 % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        # 将傅立叶系数转换为 NumPy 数组
        fourier_coeffs_np = model.kan_line.fouriercoeffs.detach().cpu().numpy()

        # 指定保存的文件路径
        file_path = 'fourier_coeffs.csv'

        # 打开文件并写入
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入多行，每行是傅立叶系数的一个切片
            writer.writerow(['Real Part', 'Imaginary Part', 'Input Dimension', 'Grid Size'])
            for outdim in range(fourier_coeffs_np.shape[1]):
                for inputdim in range(fourier_coeffs_np.shape[2]):
                    for gridsize in range(fourier_coeffs_np.shape[3]):
                        real_part = fourier_coeffs_np[0, outdim, inputdim, gridsize]
                        imaginary_part = fourier_coeffs_np[1, outdim, inputdim, gridsize]
                        writer.writerow([real_part, imaginary_part, inputdim, gridsize])

        print(f"Fourier coefficients saved to {file_path}")'''

    # 在整个批次上进行一次梯度计算和裁剪
    '''
    if isinstance(loaded_valid_loader, list):
        avg_vali_loss = 0
    else:
        model.eval()
        total_loss_val = 0.0
        vali_num = 0
        arr_data = []

        for batch_idx, data in enumerate(valid_loader):

            label_value = []
            y = data[0]
            label_value.append(torch.unsqueeze(y, dim=0))
            graph_list = update_node_features(data[1]).to(device)
            node_features = graph_list.ndata['feat'].to(device)
            #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu()
            output = model(graph_list, node_features).cpu()

            
            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)
                
                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
            
            arr_pred = arr_pred.float()
            arr_label = arr_label.float()
            loss = loss_fn(arr_pred, arr_label)
            #loss = FocalLoss(arr_pred, arr_label)

            loss = torch.sum(loss)
            total_loss_val += loss
        '''
    total_loss_val = 0.0
    print(f"Epoch {epoch}|Train Loss: {total_train_loss:.4f}| Vali Loss:{total_loss_val:.4f}")

    return total_train_loss, total_loss_val


def predicting(model, device, data_loader):
    model.eval()
    
    total_preds = torch.Tensor().cpu()
    total_labels = torch.Tensor().cpu()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            y = data[0]
            #true = inverse_min_max_normalize(y,min_val, max_val)
            
            graph_list = update_node_features(data[1]).to(device)
            #node_features = graph_list.ndata['feat'].to(device)
            node_features = add_noise(graph_list.ndata['feat'],noise=True).to(device)
            #output = model(batch_g_list = graph_list, device = device, resent = resent,pooling=pooling).cpu() 
            output = model(graph_list, node_features).cpu()

            arr_label = torch.Tensor().cpu()
            arr_pred = torch.Tensor().cpu()
            for j in range(y.shape[1]):
                c_valid = np.ones_like(y[:, j], dtype=bool)
                c_label, c_pred = y[c_valid, j], output[c_valid, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)

                arr_label = torch.cat((arr_label,c_label),0)
                arr_pred = torch.cat((arr_pred,c_pred),0)
                    
            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)

    AUC = roc_auc_score(total_labels.numpy().flatten(), total_preds.numpy().flatten())
    
    
    return AUC



def parse_arguments():
    parser = argparse.ArgumentParser(description="示例命令行工具")

    # 添加命令行参数
    parser.add_argument("--config", type=str, help="配置文件路径")

    args = parser.parse_args()
    args.config = './config/c_path.yaml'
    # 如果提供了配置文件路径，则加载配置文件
    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)

        # 将配置文件中的参数添加到命令行参数中
        for key, value in config.items():
            setattr(args, key, value)

    return args


if __name__ == '__main__':
    
    #mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # 设置种子
    seed = 42
    set_seed(seed)

    args = parse_arguments()
    for key, value in vars(args).items():
        if key != 'config':
            print(f"{key}: {value}")
    datafile = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio
    target_map = {'tox21':12,'muv':17,'sider':27,'clintox':2,'bace':1,'bbbp':1,'hiv':1}
    target_dim = target_map[datafile]

    

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond

    encode_dim = [0,0]
    encode_dim[0] = 92
    encode_dim[1] = 21
    

    
    creat_data(datafile, encoder_atom, encoder_bond, batch_size, train_ratio, vali_ratio, test_ratio)

    model_select = args.model_select
    loss_sclect = args.loss_sclect


    # 加载 DataLoader 使用的数据集和其他必要信息
    state = torch.load('data/processed/'+datafile+'.pth')

    # 重新创建 CustomDataset 和 DataLoader
    loaded_train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])
    
   

    loaded_train_loader = DataLoader(loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)
    if vali_ratio == 0.0:
        loaded_valid_loader = []
    else:
        loaded_valid_loader = DataLoader(loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    loaded_test_loader = DataLoader(loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn)


    print('dataset was loaded!')

    print("length of training set:",len(loaded_train_dataset))
    print("length of validation set:",len(loaded_valid_dataset))
    print("length of testing set:",len(loaded_test_dataset))
    
    iter = args.iter
    LR = args.LR
    NUM_EPOCHS = args.NUM_EPOCHS
    grid_feat = args.grid_feat
    num_layers = args.num_layers
    pooling = args.pooling

    All_AUC = []
    for i in range(iter):
        
        AUC_list = []
        if model_select == 'ka_gnn':
            model = KA_GNN(in_feat=128, hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'ka_gnn_two':
            model = KA_GNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)
        
        elif model_select == 'mlp_sage':
            model = MLPGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'mlp_sage_two':
            model = MLPGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                               grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage':
            model = KANGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)

        elif model_select == 'kan_sage_two':
            model = KANGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=target_dim, 
                           grid_feat=grid_feat, num_layers=num_layers, pooling = pooling, use_bias=True)
                
        
        #print(model)
        # 统计模型的总参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        train_loss_dic = {}
        vali_loss_dic = {}

        #model = modeling().to(device)
        model = model.to(device)
        if loss_sclect == 'l1':
            #loss_fn = nn.L1Loss()
            loss_fn = nn.L1Loss(reduction='sum')#sum，mean,none

        elif loss_sclect == 'l2':
            loss_fn = nn.MSELoss(reduction='none')

        elif loss_sclect == 'sml1':
            loss_fn = nn.SmoothL1Loss(reduction='sum')#mean,none,sum

        elif loss_sclect == 'bce':
            loss_fn = nn.BCELoss(reduction='mean')
        
        else:
            print('No Found the Loss function!')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        best_auc = 0
        for epoch in range(NUM_EPOCHS):
            train_loss,vali_loss = train(model, device, loaded_train_loader, loaded_valid_loader, optimizer, epoch + 1)

            
            AUC = predicting(model, device, loaded_test_loader)
            
            
            if AUC > best_auc:
                best_auc = AUC
                logger.info(f'AUC: {best_auc:.5f}')
                formatted_number = "{:.5f}".format(best_auc)
                best_auc = float(formatted_number)
                AUC_list.append(best_auc)

                print(f"Epoch [{epoch+1}], Learning Rate: {scheduler.get_last_lr()}")

                
                #if epoch+1 % 1000 == 0:
                #print(f'Epoch {epoch}, Loss: {loss.item()}')
                # 将傅立叶系数转换为 NumPy 数组
                fourier_coeffs_np = model.kan_line.fouriercoeffs.detach().cpu().numpy()

                # 指定保存的文件路径
                file_path = 'fourier_coeffs.csv'

                # 打开文件并写入
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # 写入多行，每行是傅立叶系数的一个切片
                    writer.writerow(['Real Part', 'Imaginary Part', 'Input Dimension', 'Grid Size'])
                    for outdim in range(fourier_coeffs_np.shape[1]):
                        for inputdim in range(fourier_coeffs_np.shape[2]):
                            for gridsize in range(fourier_coeffs_np.shape[3]):
                                real_part = fourier_coeffs_np[0, outdim, inputdim, gridsize]
                                imaginary_part = fourier_coeffs_np[1, outdim, inputdim, gridsize]
                                writer.writerow([real_part, imaginary_part, inputdim, gridsize])

                print(f"Fourier coefficients saved to {file_path}")
        
            if epoch % 10 == 0:
                #MAE_list.append(best_MAE)
                print("-------------------------------------------------------")
                print("epoch:",epoch)
                print('best_MAE:', best_auc)
            
            if epoch == NUM_EPOCHS-1:
                print(f"the best result up to {i+1}-loop is {best_auc:.4f}.")
                formatted_number = "{:.5f}".format(best_auc)
                All_AUC.append(best_auc)
    torch.save(model.state_dict(), 'model.pth')
    
    # 计算均值
    mean_value = statistics.mean(All_AUC)
    # 计算标准差
    std_dev = statistics.stdev(All_AUC)
    # 打印结果
    print("均值:", mean_value)
    print("标准差:", std_dev)
