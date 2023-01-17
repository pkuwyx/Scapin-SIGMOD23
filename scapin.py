import heapq

import numpy
import numpy as np
import scipy.sparse
import torch
from numba import njit, prange

import utils
from models import lpa, gcn


class Scapin:
    @staticmethod
    def generate(adj: scipy.sparse.spmatrix, split_idx: dict,
                 ptb_rate: float, th: float) -> scipy.sparse.lil_matrix:
        """
        Generate attacked graph using Scapin method.

        Args:
            adj: the original adjacency matrix in scipy sparse matrix format
            split_idx: built-in dictionary with "train", "valid", "test" as its key,
                       each containing a built-in list of idx
            ptb_rate: built-in float number (0, 1), the perturb rate
            th: built-in float number, the parameter of Scapin method

        Returns:
            the perturbed adjacency matrix in scipy sparse lil format
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        worker = Scapin.ScapinExec(adj=adj, split_idx=split_idx,
                                   _ptb_rate=ptb_rate, _th=th, _device=device)
        return worker.attack()

    @staticmethod
    def evaluate(adj: scipy.sparse.spmatrix, features: numpy.ndarray, labels: numpy.ndarray, split_idx: dict,
                 model: str) -> float:
        """
        Evaluate a graph on a specific node classification method once.

        Args:
            adj: the adjacency matrix in scipy sparse matrix format
            features: 2D numpy array, i-th row representing the feature of i-th node
            labels: numpy array
            split_idx: built-in dictionary with "train", "valid", "test" as its key,
                       each containing a built-in list of idx
            method: built-in string, 'LPA' or 'GCN', representing the method, default='LBA'
            seed: integer, the random seed for evaluation, default=9

        Returns:
            the accuracy metric
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        features_torch = torch.tensor(features)
        labels_torch = torch.tensor(labels)
        if model == "LPA":
            m_r = lpa
        elif model == "GCN":
            m_r = gcn
        else:
            raise NotImplementedError
        res = m_r.test(utils.sparse_mx_to_torch_sparse_tensor(adj), features_torch,
                         labels_torch, split_idx, device)
        return res

    class ScapinExec:
        def __init__(self, adj, split_idx, _ptb_rate, _device, _th):
            self.node_cnt_score = None
            self.node_degree = None
            self.adj_norm_lil = None
            self.degree_heap = None
            self.node_cnt_sum = None
            self.receptive_vector = None
            self.num_ones = None
            self.adj_lil = None
            self.init_activate_value = None  # numpy array
            self.ptb_rate = _ptb_rate
            self.device = _device
            self.train_idx = split_idx["train"]  # numpy array
            self.valid_idx = split_idx["valid"]  # numpy array
            self.test_idx = split_idx["test"]  # numpy array
            self.adj_sp = adj.tocoo()  # scipy.sparse coo matrix
            self.num_nodes = self.adj_sp.get_shape()[0]  # int, # of nodes
            self.adj_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_sp)  # torch sparse coo matrix
            self.adj_norm_sp = utils.aug_normalized_adjacency(self.adj_sp)  # scipy.sparse coo matrix
            self.adj_norm_torch = utils.sparse_mx_to_torch_sparse_tensor(self.adj_norm_sp)  # torch sparse coo matrix
            degree_torch = torch.sparse.sum(self.adj_torch, dim=0).values()
            self.deg = np.array(degree_torch)  # numpy array
            self.ptb_num = int(self.deg.sum() // 2 * self.ptb_rate)
            self.th = _th

        def __get_min_unconnected_node(self, heap_, selected_node):
            tmp_list = []
            res_node = 0
            while True:
                tmp_node = heapq.heappop(heap_)
                tmp_list.append(tmp_node)
                if self.adj_lil[selected_node, tmp_node[1]] == 0:
                    res_node = tmp_node[1]
                    break
            for i in range(len(tmp_list)):
                heapq.heappush(heap_, tmp_list[i])
            return res_node

        @staticmethod
        @njit
        def __try_add_edge(node_cnt_sum, K, col, data, th):
            delta = 0
            for i in range(len(col)):
                if node_cnt_sum[col[i]] <= th < K * data[i]:
                    delta = delta + 1
            return delta

        @staticmethod
        def __exec_add_edge(node_cnt_sum, K, col, data):
            for tx in prange(col.size):
                if node_cnt_sum[col[tx]] < K * data[tx]:
                    node_cnt_sum[col[tx]] = K * data[tx]

        @staticmethod
        def __prep_cnt_sum(node_cnt_sum, K_arr, row, col, data):
            for tx in prange(col.size):
                if node_cnt_sum[col[tx]] < K_arr[row[tx]] * data[tx]:
                    node_cnt_sum[col[tx]] = K_arr[row[tx]] * data[tx]

        def __get_min_unconnected_node_and_add(self, heap_, selected_node):
            tmp_list = []
            res_node = 0
            tmp_node = 0
            while True:
                tmp_node = heapq.heappop(heap_)
                if self.adj_lil[selected_node, tmp_node[1]] == 1:
                    tmp_list.append(tmp_node)
                if self.adj_lil[selected_node, tmp_node[1]] == 0:
                    res_node = tmp_node[1]
                    tmp_list.append((tmp_node[0] + 1, tmp_node[1]))
                    break
            for i in range(len(tmp_list)):
                heapq.heappush(heap_, tmp_list[i])
            return res_node

        def __get_max_score_node(self, heap_):
            max_node = 0
            max_node_score = 0
            tmp_node1 = heapq.heappop(heap_)
            while True:
                from_node = self.__get_min_unconnected_node(self.degree_heap, tmp_node1[2])
                cur_row_coo = self.adj_norm_lil[tmp_node1[2]].tocoo()
                update_tmp_node1_score = - self.__try_add_edge(self.node_cnt_sum,
                                                               1 / (self.node_degree[from_node] + 1),
                                                               cur_row_coo.col, cur_row_coo.data, self.th)
                count = - update_tmp_node1_score + self.node_cnt_score
                tmp_node2 = heapq.heappop(heap_)
                if update_tmp_node1_score > tmp_node2[0]:
                    heapq.heappush(heap_, (update_tmp_node1_score, tmp_node1[1], tmp_node1[2]))
                    tmp_node1 = tmp_node2
                if update_tmp_node1_score <= tmp_node2[0]:
                    heapq.heappush(heap_, tmp_node2)
                    max_node = tmp_node1[2]
                    max_node_score = count
                    break
            return max_node, max_node_score

        def attack(self):
            self.node_cnt_sum = np.zeros(self.num_nodes)
            self.degree_heap = []
            score_heap = []
            self.adj_lil = self.adj_sp.tolil()
            self.adj_norm_lil = self.adj_norm_sp.tolil()
            idx_train_list = self.train_idx.tolist()
            self.num_ones = np.ones(self.num_nodes)
            node_cnt = np.array(self.adj_norm_lil[idx_train_list].tocsr().max(axis=0).todense()).flatten()
            for i in idx_train_list:
                node_cnt[i] = 1
            self.node_degree = torch.sparse.sum(self.adj_torch.to(self.device), [0]).to_dense().cpu().numpy()

            for i in idx_train_list:
                heapq.heappush(self.degree_heap, (self.node_degree[i], i))

            self.__prep_cnt_sum(self.node_cnt_sum, node_cnt, self.adj_norm_sp.row, self.adj_norm_sp.col,
                                self.adj_norm_sp.data)

            self.receptive_vector = (self.node_cnt_sum > self.th) + 0
            self.node_cnt_score = self.num_ones.dot(self.receptive_vector)
            TAT = np.arange(self.num_nodes)
            for _ in range(self.num_nodes):
                i = TAT[_]
                from_node = self.__get_min_unconnected_node(self.degree_heap, i)
                cur_row_coo = self.adj_norm_lil[i].tocoo()
                delta = self.__try_add_edge(self.node_cnt_sum,
                                            1 / (self.node_degree[from_node] + 1),
                                            cur_row_coo.col, cur_row_coo.data, self.th)
                heapq.heappush(score_heap, (-delta, _, i))

            num_count = 0
            add_res_list = []
            while True:
                to_node, self.node_cnt_score = self.__get_max_score_node(score_heap)
                from_node = self.__get_min_unconnected_node_and_add(self.degree_heap, to_node)
                self.node_degree[from_node] += 1
                self.adj_lil[from_node, to_node] = 1
                self.adj_lil[to_node, from_node] = 1
                num_count += 1
                cur_row_coo = self.adj_norm_lil[to_node].tocoo()
                self.__exec_add_edge(self.node_cnt_sum, 1 / self.node_degree[from_node], cur_row_coo.col,
                                     cur_row_coo.data)
                if num_count >= self.ptb_num:
                    break

            return self.adj_lil.copy()
