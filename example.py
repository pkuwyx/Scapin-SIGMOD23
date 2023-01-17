from deeprobust.graph.data import Dataset
from scapin import Scapin
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--ptb_rate", default=0.03, type=float)
parser.add_argument("--dataset", default="cs")
parser.add_argument("--th", default=-1, type=float)
parser.add_argument("--model", default="LPA", type=str)

args, unknown = parser.parse_known_args()

th = utils.check_th(args.dataset, args.th)
ptb_rate = args.ptb_rate
model = args.model

print(f"EXP:\ndataset = {args.dataset}\nperturbation rate = {ptb_rate}")
print(f"threshold = {th}\nevaluation method = {model}")

if args.dataset in ["cora", "citeseer", "cora_ml"]:
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj = data.adj
    features = np.asarray(data.features.todense())
    labels = data.labels
    num_nodes = len(labels)
elif args.dataset in ["cs", "phy"]:
    adj, features, labels = utils.load_npz_to_sparse_graph(f"ms_dataset/ms_academic_{args.dataset}.npz")
    features = np.asarray(features.todense())
    num_nodes = len(labels)
else:
    raise NotImplementedError

train_idx, valid_idx, test_idx = utils.get_train_val_test(num_nodes, val_size=0.1, test_size=0.8, stratify=labels)

split_idx = {
    "train": train_idx,
    "valid": valid_idx,
    "test": test_idx
}

res = Scapin.evaluate(adj=adj, features=features, labels=labels, split_idx=split_idx, model=model)

print(f"original acc: {res}")

perturbed_adj = Scapin.generate(adj=adj, split_idx=split_idx, ptb_rate=ptb_rate, th=th)

perturbed_res = Scapin.evaluate(adj=perturbed_adj, features=features, labels=labels, split_idx=split_idx,
                                model=model)

print(f"perturbed acc: {perturbed_res}")
