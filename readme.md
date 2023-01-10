# Scapin: Scalable Graph Perturbation by Augmented InfluenceMaximization

The repository contains the implementation of our proposed Scapin method.

**To set up the environment:**

```bash
conda create -n scapin python=3.9
pip install -r requirements.txt
```

**The library provides two main methods: `generate` (perturbation generation) and `evaluate` (node classification evaluation).**

```python
Scapin.generate(adj: scipy.sparse.spmatrix, features: numpy.ndarray, labels: numpy.ndarray, split_idx: dict,
                ptb_rate: float, th: float, seed: int, device: str) -> scipy.sparse.lil_matrix:
```

Generate attacked graph using Scapin method.

Args:
    adj: the original adjacency matrix in scipy sparse matrix format
    features: 2D numpy array, i-th row representing the feature of i-th node
    labels: numpy array
    split_idx: built-in dictionary with "train", "valid", "test" as its key,
               each containing a built-in list of idx
    ptb_rate: built-in float number (0, 1), the perturb rate
    th: built-in float number, the parameter of Scapin method
    seed: built-in integer, the random seed
    device: built-in string, the pytorch device, default='cpu',
            see https://pytorch.org/docs/stable/tensor_attributes.html#torch.device

Returns:
    the perturbed adjacency matrix in scipy sparse lil format

```python
Scapin.evaluate(adj: scipy.sparse.spmatrix, features: numpy.ndarray, labels: numpy.ndarray, split_idx: dict,
                method: str, seed: int, device: str) -> float:
```

Evaluate a graph on a specific node classification method once.

Args:
    adj: the adjacency matrix in scipy sparse matrix format
    features: 2D numpy array, i-th row representing the feature of i-th node
    labels: numpy array
    split_idx: built-in dictionary with "train", "valid", "test" as its key,
               each containing a built-in list of idx
    method: built-in string, 'LPA' or 'GCN', representing the method, default='LBA'
    seed: integer, the random seed for evaluation, default=9
    device: built-in string, the pytorch device, default='cpu',
            see https://pytorch.org/docs/stable/tensor_attributes.html#torch.device

Returns:
    the accuracy metric

**See `example.py` for a detailed example with sample data loader.**

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

