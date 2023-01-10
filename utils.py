import torch
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
import torch.nn.functional as F


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


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


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    from sklearn.model_selection import train_test_split
    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels


def test(adj, features, labels, train_idx, val_idx, test_idx, device):
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, train_idx)
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()


def check_th(dataset, th):
    if th != -1 and type(th) == float:
        return th
    elif th == -1:
        if dataset == 'cora':
            return 0.002
        elif dataset == 'cora_ml':
            return 0.084
        elif dataset == 'citeseer':
            return 0.05
        elif dataset == 'cs':
            return 0.07
        else:
            raise NotImplementedError
    else:
        raise AttributeError
