from deeprobust.graph.data import Dataset
from scapin import Scapin
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--dataset", default="citeseer")

args, unknown = parser.parse_known_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')

worker = Scapin(adj=data.adj, features=np.asarray(data.features.todense()), labels=data.labels,
                split_idx={"train": data.idx_train,
                           "valid": data.idx_val,
                           "test": data.idx_test},
                _ptb_rate=0.05, _device="cuda:0", _th=0.085)

adj = worker.attack()

print(adj)
