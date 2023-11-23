import torch
import torch.nn as nn
import torch.nn.functional as F
# from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit
from torch.utils.data import Dataset
from collections import deque


# 后续考虑增量式增加数据，而不是all in
class FitDataset2D_train(Dataset):
    def __init__(self, data_tensor):
        self.y_data = torch.log(data_tensor.T[0])
        self.x_data = data_tensor.T[1:].T.view(data_tensor.shape[0], 12, -1)
        self.len = data_tensor.shape[0]

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()  # 继承Net到nn.module

        # 定义层，nn.Linear(in_features,out_features),Linear第一个参数都是层左边输入的特征数，第二个参数是层右边输出的特征数
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 6))
        self.conv2 = torch.nn.Conv2d(4, 4, (2, 3))
        self.conv3 = torch.nn.Conv2d(4, 4, (1, 1))

        self.pool1 = torch.nn.MaxPool2d((2, 5))
        self.pool2 = torch.nn.MaxPool2d((4, 11))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            if type(x) == list:
                x_batch_size = len(x)
                x = torch.tensor(x).view(x_batch_size, 12, -1)
        x = x[:, None, :, :]
        x1 = self.relu(self.conv1(x))
        x1_p = self.pool1(x1)
        x2 = self.relu(self.conv2(x1_p))
        x2_p = self.pool2(x2)
        x3 = self.conv3(x2_p).view(x.shape[0], -1)
        # output = x3.max(dim=1)[0]
        output = torch.mean(x3, dim=1)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


class SurrogateFCN141:
    def __init__(self, buffer_len, is_load=False, load_path=None):
        self.model = Model()  # 代理模型
        if is_load:
            assert load_path is not None
            self.model.load_state_dict(torch.load(load_path))
            print('Surrogate has loaded model from {}'.format(load_path))
        self.buffer = deque(maxlen=buffer_len)  # 存放数据的，格式为[fitness, var]
        # ------------------------- 训练参数 ---------------------------
        self.lr_first = 0.06
        self.lr_increm = 0.03
        self.batch_size = 32
        self.EPOCH_first = 12
        self.EPOCH_increm = 4
        self.optimizer_first = torch.optim.Adam(self.model.parameters(), lr=self.lr_first)  # 优化器
        self.optimizer_increm = torch.optim.Adam(self.model.parameters(), lr=self.lr_increm)
        self.loss_func = torch.nn.MSELoss()

        self.train_step = 0

    def add_samples(self, data):
        for eve in data:
            if eve[0] == 'GT':
                self.buffer.append(eve[1:])

    def get_buffer_size(self):
        return len(self.buffer)

    def train_model(self, isfirst, istrain=True):  # if is first the epoch will be longer
        # istrain == False 则说明只计算loss不真训练
        train_data = torch.tensor(self.buffer)
        current_dataset = FitDataset2D_train(train_data)
        current_dataloader = DataLoader(dataset=current_dataset, batch_size=self.batch_size)
        losses = []
        if isfirst:
            EPOCH = self.EPOCH_first
            optimizer = self.optimizer_first
        else:
            EPOCH = self.EPOCH_increm
            optimizer = self.optimizer_increm
        for epoch in range(EPOCH):
            loss_save = []
            for i, data in enumerate(current_dataloader):
                x, y = data
                y_hat = self.model(x)
                loss = self.loss_func(y_hat, y)
                if istrain:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_save.append(loss.item())
            losses.append(sum(loss_save) / len(loss_save))
            if not istrain:  # 不训练的话计算一个epoch的loss就够了
                break
        # print('Training step of {} has epoch losses of {}'.format(self.train_step, losses))
        self.train_step += 1
        return losses

    def save_model(self, root_path, generation):
        torch.save(self.model.state_dict(), root_path + '-model-' + str(generation) + '.pt')

    def load_model(self, torch_path):
        self.model.load_state_dict(torch.load(torch_path))
        print('Pretrained model of ', torch_path, ' has been loaded !!')

