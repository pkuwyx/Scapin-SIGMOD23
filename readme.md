# Scapin: Scalable Graph Perturbation by Augmented InfluenceMaximization

The repository contains the implementation of our proposed Scapin method.

System software requirement:

- Ubuntu 16.04 or higher
- `g++` 5.4.0 or higher
- `conda` 4.11.0 or higher, the package manager for Python which installs all other dependencies
- For GPU acceleration, `cuda` 10.2 or higher and the corresponding version of PyTorch 1.8.1 or higher

You may install `g++` by executing `apt install g++`

You may install `conda` by following the instructions on https://docs.conda.io/projects/miniconda/en/latest/index.html#quick-command-line-install

To set up the environment:

```bash
conda create -n scapin python=3.9
conda activate scapin
pip install -r requirements.txt
pip install torchmetrics==1.0
```

You may need to install the cuda-enabled version of PyTorch manually for GPU acceleration.

The library provides two main methods: `generate` (perturbation generation) and `evaluate` (node classification evaluation).

```python
Scapin.generate(adj: scipy.sparse.spmatrix, split_idx: dict, ptb_rate: float, th: float) -> scipy.sparse.lil_matrix:
```

Generate attacked graph using Scapin method.

Args:
    adj: the original adjacency matrix in scipy sparse matrix format
    split_idx: built-in dictionary with "train", "valid", "test" as its key,
               each containing a built-in list of idx
    ptb_rate: built-in float number (0, 1), the perturb rate
    th: built-in float number, the parameter of Scapin method

Returns:
    the perturbed adjacency matrix in scipy sparse lil format

```python
Scapin.evaluate(adj: scipy.sparse.spmatrix, features: numpy.ndarray, labels: numpy.ndarray, split_idx: dict, method: str) -> float:
```

Evaluate a graph on a specific node classification method once.

Args:
    adj: the adjacency matrix in scipy sparse matrix format
    features: 2D numpy array, i-th row representing the feature of i-th node
    labels: numpy array
    split_idx: built-in dictionary with "train", "valid", "test" as its key,
                       each containing a built-in list of idx
    model: built-in string, 'LPA' or 'GCN', representing the model, default='LBA'

Returns:
    the accuracy metric

To use the library:

```python
from scapin import Scapin

[load data]

ptb_adj = Scapin.generate(adj, features, labels, split_idx, ptb_rate, th, seed, device)
Scapin.evaluate(adj, features, labels, split_idx, method, seed, device)
Scapin.evaluate(ptb_adj, features, labels, split_idx, method, seed, device)
```

See `example.py` for a detailed example with sample data loader.

```bash
python example.py
```

## Results

1. Effectiveness evaluation

![image-20221014204334860](fig/effectiveness1.png)

![image-20221014204348834](fig/effectiveness2.png)

2. Runtime comparison

![image-20221014204407649](fig/runtime.png)

3. Memory comparison

![image-20221014204419576](fig/memory.png)

4. Interpretability

![image-20221014204500189](fig/interpretability.png)

5. Transferability

![image-20221014204447423](fig/transferability.png)

6. Sensitivity Analyses

![image-20221014204535325](fig/sensitivity.png)

6. Defense

![image-20221014204544589](fig/defense.png)

