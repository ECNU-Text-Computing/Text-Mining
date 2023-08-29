import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data


def encode_onehot(labels):
    '''
    先将所有标签用set存储，得到无重复数据集
    为每个标签分配一个编号，创建单位矩阵，每一行代表一个label
    每个数据对应的标签表示成one-hot向量
    '''
    # set()无序不重复元素集
    classes = set(labels)
    # no.identity 创建一个单位矩阵
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # get函数得到字典key对应的value,字典key是label,value是矩阵的每一行
    # map()做映射
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(opt):
    """加载数据集"""
    print('Loading {} dataset...'.format(opt.network))

    # content file的每一行格式为 <paper_id> <word_attributes> <class_label> 分别对应0，1：-1，-1
    idx_features_labels = np.genfromtxt("./data/{}.content".format(opt.network), dtype=np.dtype(str))
    # sp.csr_matrix 存储csr型稀疏性矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

    # 重新排列文件中节点顺序 {old_id: number}
    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered直接从cora.cites文件中读取结果， shape: edge_num x 2
    edges_unordered = np.genfromtxt("./data/{}.cites".format(opt.network), dtype=np.int32)
    # 将edges_unordered的old id换成number
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 构建对称的邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 特征矩阵归一化
    features = normalize(features)
    # 邻接矩阵归一化
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # {标签：索引列表并随机打乱}
    labels_map = {i: [] for i in range(labels.shape[1])}
    labels = np.where(labels)[1]  # 第一个非零元素的列索引
    for i in range(labels.shape[0]):
        labels_map[labels[i]].append(i)
    for ele in labels_map:
        random.shuffle(labels_map[ele])

    # 划分数据集
    idx_train = list()
    idx_val = list()
    idx_test = list()
    for ele in labels_map:
        label_sample_count = len(labels_map[ele])  # 每个标签对应的样本数量
        idx_train.extend(labels_map[ele][0:int(opt.train_rate * label_sample_count)])
        idx_val.extend(
            labels_map[ele][int(opt.train_rate * labels[0]):int((opt.train_rate + opt.val_rate) * label_sample_count)])
        idx_test.extend(labels_map[ele][int((opt.train_rate + opt.val_rate) * label_sample_count):])

    # 类型为torchTensor是为了在神经网络中的乘法
    features = torch.FloatTensor(np.array(features.todense()))
    # 将稀疏矩阵转换为FloatTensor
    adj = torch.FloatTensor(np.array(adj.todense()))

    # 获得每个节点的邻居{v0:[v0的邻居集合],v1:[v1的邻居集合]}
    adj_lists = defaultdict(set)  # 工厂函数是set，当尝试访问一个字典中不存在的键时，会自动创建一个空集合作为默认值4
    rows, cols = np.where(adj != 0)
    for row, col in zip(rows, cols):
        adj_lists[row].add(col)
        adj_lists[col].add(row)
    # 断言检查：节点数相等
    assert len(features) == len(labels) == len(adj_lists)

    return adj, features, labels, idx_train, idx_val, idx_test, adj_lists


def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)  # 构造对角元素是r_inv的对角矩阵
    mx = r_mat_inv.dot(mx)
    return mx


class Dataload(data.Dataset):

    def __init__(self, labels, id):
        self.data = id
        self.labels = labels

    def __getitem__(self, index):
        return index, self.labels[index]

    def __len__(self):
        return self.data.__len__()
