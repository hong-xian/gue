# AAAI24 [paper](https://arxiv.org/abs/2401.17523)
This is the official repository for "Game-Theoretic Unlearnable Example Generator"). This repository contains an implementation of GUE and evaluation on poisoned datasets.
## Requirements:  
* Python 3.11.3
* PyTorch 2.0.1
* Torchvision 2.0.0


## Running experiments:  
We give an example of training GUE generator on CIFAR-10 dataset:
 ```
python gue.py --dataset cifar10 --epochs 50 --eta 1.5 --lr_atk 0.1  --lr_cls 0.01 --tensorboard_path save/path
 ```


Evaluate the unlearnable examples generated by GUE generator：
 ```
python evaluation.py --dataset cifar10 --arch ResNet18 --out_dir ./results --path ./your/path/atkmodel --exp_name gue  --train_loss ST  --lr 0.01
 ```

We also provide pretrained GUE generator. To test it, use:
 ```
python evaluation.py --dataset cifar10 --arch ResNet18 --out_dir ./results --path ./gue_cifar10.pth --exp_name gue  --train_loss ST  --lr 0.01
 ```
