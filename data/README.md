This project uses the same data format as GEM, which includes benchmark such as Permutation MNIST, rotation MNIST, split CIFAR, etc.
To prepare the datasets, follow the [GEM's instruction](https://github.com/facebookresearch/GradientEpisodicMemory) to create the `mnist_permutations.pt` and `cifar100.pt` in the `data/raw/` folder.
Then, run the `data/cifar100.py` and `data/mnist_permutations.py` scripts to create the corresponding benchmarks. Each benchmark will consists of two files: the `-val.pt` file only contains 3 tasks used for hyper-parameter cross-validation and the `-cl.pt` file contains the remaining tasks for actual continual learning.

