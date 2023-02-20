import torch
import argparse


def init():
    parser = argparse.ArgumentParser('parameters for train')

    # optimize
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size for model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for model')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs for train')
    # model
    parser.add_argument('--model_name', type=str, default='cnn', help='model for train')

    args = parser.parse_args()
    return args


def train_one_epoch():
    pass


def train(args):
    # 获取数据集
    # 数据处理
    # 训练
    for epoch in range(args.epochs):
        train_one_epoch()
    # 保存数据

    pass



