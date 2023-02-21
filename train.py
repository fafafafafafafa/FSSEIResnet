import torch
import torch.utils.data
import torchinfo
import argparse
import MyDataset
import MyModel


def init():
    parser = argparse.ArgumentParser('parameters for train')
    # dataset
    parser.add_argument('--train_dataset_num', type=int, default=90, help='num of train class')
    parser.add_argument('--test_dataset_num', type=list, default=[10, 20, 30], help='num of train class')

    # parameters
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size for model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for model')
    parser.add_argument('--epochs', type=int, default=100, help='max epochs for train')

    # model
    parser.add_argument('--model_name', type=str, default='resnet18', help='model for train')

    # optimizer
    parser.add_argument('--optimizer_name', type=str, default='adam', help='optimizer for train')

    args = parser.parse_args()
    return args


def train_one_epoch():
    pass


def train():
    # 初始化
    args = init()
    print(args.train_dataset_num)
    # 获取数据集
    train_dataset = MyDataset.get_train_dataset(args.train_dataset_num)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # model
    # my_model = None
    if args.model_name == 'resnet18':
        my_model = MyModel.resnet18()
    else:
        raise ValueError('model is None!')
    print(torchinfo.summary(my_model))
    # optimizer
    # optimizer = None
    if args.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(my_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        raise ValueError('optimizer is None!')
    # 数据处理
    # 训练
    for epoch in range(args.epochs):
        train_one_epoch()
    # 保存数据

    pass


if __name__ == '__main__':
    train()

