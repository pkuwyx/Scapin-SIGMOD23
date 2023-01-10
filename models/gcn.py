from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
import torch.nn.functional as F
import torch

hidden = 16
dropout = 0.5


def test(adj, features, labels, split_idx, device, val_idx):
    gcn = GCN(nfeat=features.shape[1],
              nhid=hidden,
              nclass=labels.max().item() + 1,
              dropout=dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, split_idx["train"])
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[split_idx["test"]], labels[split_idx["test"]])
    acc_test = accuracy(output[split_idx["test"]], labels[split_idx["test"]])
    return acc_test
