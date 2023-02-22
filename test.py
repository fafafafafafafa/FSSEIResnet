import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import initializion
import MyDataset
import MyModel
import ResNetRadio
import torchinfo


def normalize_data(X):
    min_value = X.min()
    max_value = X.max()
    X = (X - min_value) / (max_value - min_value)
    X = np.float32(X)
    return X


def test():
    args = initializion.init()
    # 获取测试集
    test_dataset = MyDataset.get_val_dataset(90, normalize_data)
    # test_dataset = MyDataset.get_train_dataset(90, normalize_data)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if args.model_name == 'resnet18':
        my_model = MyModel.ResNetForClass(ResNetRadio.resnet18(), args)
    elif args.model_name == 'resnet34':
        my_model = MyModel.ResNetForClass(ResNetRadio.resnet34(), args)
    else:
        raise ValueError('model is None!')
    torchinfo.summary(my_model)
    print(torch.cuda.is_available())

    if args.loss_fn_name == 'cross_entropy_loss':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError('loss_fn is None!')

    # my_model.load_state_dict(torch.load(args.early_stopping_dir_path+args.model_name))
    my_model.load_state_dict(torch.load(args.early_stopping_dir_path
                                        +args.model_name+'/'+'checkpoint.pt')["model_state_dict"])

    if torch.cuda.is_available():
        my_model = my_model.cuda()

    my_model.eval()

    running_loss = 0.0
    running_correct = 0
    test_loss = 0

    for batch, (X, y) in enumerate(test_data_loader, 1):
        if torch.cuda.is_available():
            # 获取输入数据X和标签Y并拷贝到GPU上
            # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
            # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
            # 但这里我们的X和y不需要进行更新，因此也不用求梯度

            signals, labels = X.cuda(), y.cuda()

        else:
            signals, labels = X, y
        with torch.no_grad():
            pred = my_model(signals)

            loss = loss_fn(pred, labels)

        # 计算一个批次的损失值和
        running_loss += loss.detach().item()
        # 计算一个批次的预测正确数
        _, labels_pred = torch.max(pred.detach(), dim=1)
        running_correct += torch.sum(labels_pred.eq(labels))

        # 打印训练结果
        if batch == len(test_data_loader):
            test_loss = running_loss / batch
            acc = 100 * running_correct / (args.batch_size * batch)
            print(
                'Batch {batch}/{iter_times},Val Loss:{loss:.4f},Va; Acc:{correct}/{lens}={acc:.4f}%'.format(
                    batch=batch,
                    iter_times=len(test_data_loader),
                    loss=running_loss / batch,
                    correct=running_correct,
                    lens=args.batch_size * batch,
                    acc=acc
                ))


if __name__ == '__main__':
    test()



