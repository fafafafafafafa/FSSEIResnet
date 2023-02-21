import numpy as np
import torch.utils.data


class MyFsSeiDataset(torch.utils.data.Dataset):

    def __init__(self, filepath_x, filepath_y, transform_x=None, transform_y=None):
        super(MyFsSeiDataset, self).__init__()
        self.X = np.load(filepath_x)
        self.Y = np.load(filepath_y)
        self.transform_x = transform_x
        self.transform_y = transform_y
        if self.transform_x:
            self.X = self.transform_x(self.X)
        if self.transform_y:
            self.Y = self.transform_y(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.Y)


def get_train_dataset(num, transform_x=None, transform_y=None):
    filepath_x = 'D:/Demo/myPycharm/FS-SEI_4800/Dataset/X_train_{}Class.npy'.format(num)
    filepath_y = 'D:/Demo/myPycharm/FS-SEI_4800/Dataset/Y_train_{}Class.npy'.format(num)
    return MyFsSeiDataset(filepath_x, filepath_y, transform_x, transform_y)


def get_test_dataset(num, transform_x=None, transform_y=None):
    filepath_x = 'D:/Demo/myPycharm/FS-SEI_4800/Dataset/X_test_{}Class.npy'.format(num)
    filepath_y = 'D:/Demo/myPycharm/FS-SEI_4800/Dataset/Y_test_{}Class.npy'.format(num)
    return MyFsSeiDataset(filepath_x, filepath_y, transform_x, transform_y)


if __name__ == '__main__':
    train_dataset = get_train_dataset(90)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    x, y = next(iter(train_dataloader))
    print("x.size: ", x.size())
    print("y.size: ", y.size())