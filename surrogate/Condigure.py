import datetime
now = datetime.datetime.now()
from torch.utils.data import Dataset
import torch

is_collect = False
buffer_path = './buffers/buffer-' + str(now.now()).split('.')[0].replace(":", "-") + '.txt'

selected_path = '../buffers/buffer-10000-GA-DE.txt'  # 所用的数据集
training_data_ratio = 0.7  # 7 : 3 = 训练 : 测试

def load_data():
    xs = []
    ys = []
    with open(selected_path, 'r') as f:
        for eve in f:
            data = eve.split(',')
            x = [float(dim) for dim in data[:-1]]
            y = float(data[-1])
            xs.append(x)
            ys.append(y)
    cutted_index = int(len(ys) * training_data_ratio)
    return xs[:cutted_index], ys[:cutted_index], xs[cutted_index:], ys[cutted_index:]


class FitDataset_train(Dataset):
    def __init__(self):
        xs, ys, _, _ = load_data()
        self.len = len(ys)
        self.x_data = torch.tensor(xs)
        self.y_data = torch.log(torch.tensor(ys))

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class FitDataset_test(Dataset):
    def __init__(self):
        _, _, xs, ys = load_data()
        self.len = len(ys)
        self.x_data = torch.tensor(xs)
        self.y_data = torch.log(torch.tensor(ys))

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class FitDataset2D_train(Dataset):
    def __init__(self):
        xs, ys, _, _ = load_data()
        self.len = len(ys)
        # 将X变成二维的12 * N的输入
        N = int(len(xs[0]) / 12)
        xs_ = []
        for each_x in xs:
            temp_x = []
            for index_ in range(12):
                temp_part = each_x[index_: index_+N]
                temp_x.append(temp_part)
            xs_.append(temp_x)
        self.x_data = torch.tensor(xs_)
        self.y_data = torch.log(torch.tensor(ys))

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class FitDataset2D_test(Dataset):
    def __init__(self):
        _, _, xs, ys = load_data()
        self.len = len(ys)
        # 将X变成二维的12 * N的输入
        N = int(len(xs[0]) / 12)
        xs_  = []
        for each_x in xs:
            temp_x = []
            for index_ in range(12):
                temp_part = each_x[index_: index_+N]
                temp_x.append(temp_part)
            xs_.append(temp_x)
        self.x_data = torch.tensor(xs_)
        self.y_data = torch.log(torch.tensor(ys))

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len