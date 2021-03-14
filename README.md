# Bilevel Continual Learning

This project contains the implementation of the paper: Bilevel Continual Learning (arXiv). 
A dual memory management strategy: a replay memory and an evaluation set. The model learns new samples by experience replay with the replay memory such that in can generalize to the evaluation set.

# Cite


# Requirements
- Pytorch 1.5.0
- CUDA 10.2

All experiments in this work was run on a single K80 GPU with 12Gb memory.

# Benchmarks
### 1. Prepare data
This project uses the same data format as GEM, which includes benchmark such as Permutation MNIST, rotation MNIST, split CIFAR, etc.
To prepare the datasets, follow the [GEM's instruction](https://github.com/facebookresearch/GradientEpisodicMemory) to create the `mnist_permutations.pt` and `cifar100.pt` in the `data/raw/` folder.
Then, run the `data/cifar100.py` and `data/mnist_permutations.py` scripts to create the corresponding benchmarks. Each benchmark will consists of two files: the `-val.pt` file only contains 3 tasks used for hyper-parameter cross-validation and the `-cl.pt` file contains the remaining tasks for actual continual learning.

### 2. Run experiments
To replicate our results on the Permuted MNIST, Split CIFAR100, Split CUB, and Split miniImagenet, run
```
chmod 777 scripts/run.sh
./scripts/run.sh
```

The results will be put in the `resuts/` folders.

### 3. Parameter Setting
The provided script `scripts/run.sh` includes the best hyper-parameter cross-validated from the cross-validation tasks. The following is the list of parameters you can experiment with

| Parameter           | Description                                                  | Values |
| :------------------ | :----------------------------------------------------------- | :-------------------------------------------------------- |
| **data_path** | path where the data will be saved | e.g. `data/` |
| **data_file** | name of the data file | e.g. `mnist_permutations.pt` |
| **use** | randomly use a subset of data. When `use < 1`, `use%` of the original data, when `use > 1`, select `use` samples from the data | e.g. `0.5` (select 50% of data), `1000` (select 1000 data samples) |
|**n_memories**| number of data stored per task | e.g. `65` |
|**memory_strength**| value of the regularizer's coefficient | e.g. `100` |
|**temperature**| temperature of the softmax in knowledge distillation | e.g. `5`|
|**lr**| (inner) learning rate | e.g. `0.1` |
|**beta**| outer learning rate (BCL) | e.g. `0.3` |
|**adapt**|  use adaptation at test time or not | e.g. `no` | 
|**adapt_lr**| learning rate of the adaptation step | e.g. `0.001` | 
|**inner_steps**| number of SGD udpates per samples | e.g. `2` | 
|**n_meta**| number of meta update per samples | e.g. `2` |
|**n_val**| percentage of the total memory used for the evaluation set (BCL-Dual) | e.g. `0.2` | 
|**replay_batch_size**| number of data in the memory used per experience replay step | e.g. `64` | 

# Acknowledgement
This project structure is based on the [GEM](https://github.com/facebookresearch/GradientEpisodicMemory) repository with additional methods, metrics and implementation improvements. 
* https://github.com/facebookresearch/GradientEpisodicMemory
