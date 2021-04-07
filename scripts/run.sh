#!/bin/bash


PREFIX="--save_path results/ --batch_size 10 --data_path data/ --cuda yes --n_epochs 1 --use 1.0 --model bcl_dual --temperature 5 --memory_strength 100"
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file mnist_permutations.pt --lr 0.03 --beta 0.3 --inner_steps 1 --n_meta 2 --n_memories 256
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file cifar-cl.pt --lr 0.3 --beta 0.1 --inner_steps 2 --n_meta 1 --n_memories 65
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file cub-cl.pt --lr 0.03 --beta 0.3 --inner_steps 1 --n_meta 2 --n_memories 50
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file mini-cl.pt --lr 0.05 --beta 0.1 --inner_steps 2 --n_meta 1 --n_memories 65
