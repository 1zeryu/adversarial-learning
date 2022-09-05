#! /bin/bash
python main.py --model resnet18 --batch_size 64 --num_workers 0 
python main.py --model resnet18 --batch_size 64 --num_workers 0

python main.py --model vit --batch_size 64 --num_workers 0 --dataset cifar10

python main.py --model vit --batch_size 128 --num_workers 4 --dataset cifar10 --lr_scheduler 