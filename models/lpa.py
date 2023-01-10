import torch
import scipy.sparse as sp
import numpy as np
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from torchmetrics.classification import Accuracy
import torch.nn.functional as F


def torch_sparse_tensor_to_sparse_mx(torch_sparse):
    """

    :type torch_sparse: torch.Tensor
    """
    torch_sparse = torch_sparse.coalesce()
    m_index = torch_sparse.indices().cpu().numpy()
    row = m_index[0]
    col = m_index[1]
    data = torch_sparse.values().cpu().numpy()
    sp_matrix = sp.coo_matrix(
        (data, (row, col)),
        shape=(torch_sparse.size()[0], torch_sparse.size()[1]))
    return sp_matrix


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0], format="coo")  # 给adj矩阵的主对角线加1
    row_sum = np.array(adj.sum(1))  # 对adj矩阵行求和，得到n*1的矩阵
    # row_sum的-1/2次方，得到一个array
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0  # 把所有的inf项变成0
    # 把d_inv_sqrt作为d_mat_inv_sqrt的对角线
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def test(adj, feature, label, split_idx, device, max_it=50):
    adj_sp = torch_sparse_tensor_to_sparse_mx(adj)
    adj_norm = aug_normalized_adjacency(adj_sp)
    adj = sparse_mx_to_torch_sparse_tensor(adj_norm)
    alpha = 0.5
    label_cnt = (torch.max(label) + 1).item()
    label_one_hot = F.one_hot(label).float()
    r_label = torch.zeros_like(label_one_hot).float()
    num_nodes = len(label)
    for i in range(num_nodes):
        r_label[i] = torch.ones(label_cnt) / label_cnt
    for source_node in split_idx["train"]:
        r_label[source_node] = label_one_hot[source_node]
    r_label = r_label.to(device)
    adj = adj.to(device)
    label_one_hot = label_one_hot.to(device)
    accuracy = Accuracy(task="multiclass", num_classes=label_cnt).to(device)
    label = label.to(device)
    ret = 0
    for it in range(max_it):
        r_label = r_label * alpha + (1-alpha) * torch.spmm(adj, r_label)
        r_label[split_idx["train"]] = label_one_hot[split_idx["train"]]
        test_acc = accuracy(r_label[split_idx["test"]], label[split_idx["test"]])
        ret = max(ret, test_acc)
    return ret
