#!/bin/bash


PREFIX="--save_path results/ --batch_size 10 --data_path data/ --cuda yes --n_epochs 1 --use 1.0 --model ctn --temperature 5 --memory_strength 100 --lr 0.01 --beta 0.05"
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file mnist_permutations.pt --inner_steps 2 --n_meta 2 --n_memories 50 --use 1000
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file cifar-cl.pt --inner_steps 2 --n_meta 2 --n_memories 50
CUDA_VISIBLE_DEVICES=0 python main.py $PREFIX --data_file mini-cl.pt  --inner_steps 2 --n_meta 2 --n_memories 50
