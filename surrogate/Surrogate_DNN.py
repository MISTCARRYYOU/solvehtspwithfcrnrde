import torch
import torch.nn as nn
import torch.nn.functional as F
# from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit
from torch.utils.data import Dataset
from collections import deque


# 后续考虑增量式增加数据，而不是all in
class FitDataset1D_train(Dataset):
    def __init__(self, data_tensor):
        self.y_data = torch.log(data_tensor.T[0])
        self.x_data = data_tensor.T[1:].T
        self.len = data_tensor.shape[0]

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class Model(nn.Module):
    """
    这是一个最简单的一层的拟合的神经网络。hidden是隐藏层，predict是预测层
    """
    def __init__(self, input_dimension):
        super(Model, self).__init__()  # 继承Net到nn.module

        # 定义层，nn.Linear(in_features,out_features),Linear第一个参数都是层左边输入的特征数，第二个参数是层右边输出的特征数
        self.hidden = nn.Linear(input_dimension, 5000)  # 一层隐藏层,n_feature个特征，n_hidden个神经元
        self.hidden2 = nn.Linear(5000, 1000)
        self.predict = nn.Linear(1000, 1)  # 预测层,n_hidden个神经元，n_output个特征的输出

    def forward(self, x):
        with torch.no_grad():
            if type(x) == list:
                x = torch.tensor(x)
        x = F.relu(self.hidden(x))  # x经过一个隐藏层，然后再被一个relu函数激活一下
        x = F.relu(self.hidden2(x))
        x = self.predict(x).view(-1)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


class SurrogateDNN:
    def __init__(self, buffer_len, units_num):
        self.model = Model(units_num)  # 代理模型
        self.buffer = deque(maxlen=buffer_len)  # 存放数据的，格式为[fitness, var]
        # ------------------------- 训练参数 ---------------------------
        self.lr = 0.05
        self.batch_size = 32
        self.EPOCH_first = 15
        self.EPOCH_increm = 5
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # 优化器
        self.loss_func = torch.nn.MSELoss()

        self.train_step = 0

    def add_samples(self, data):
        for eve in data:
            if eve[0] == 'GT':
                self.buffer.append(eve[1:])

    def get_buffer_size(self):
        return len(self.buffer)

    def train_model(self, isfirst):  # if is first the epoch will be longer
        train_data = torch.tensor(self.buffer)
        current_dataset = FitDataset1D_train(train_data)
        current_dataloader = DataLoader(dataset=current_dataset, batch_size=self.batch_size)
        losses = []
        if isfirst:
            EPOCH = self.EPOCH_first
        else:
            EPOCH = self.EPOCH_increm
        for epoch in range(EPOCH):
            loss_save = []
            for i, data in enumerate(current_dataloader):
                x, y = data
                y_hat = self.model(x)
                loss = self.loss_func(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_save.append(loss.item())
            losses.append(sum(loss_save) / len(loss_save))
        # print('Training step of {} has epoch losses of {}'.format(self.train_step, losses))
        self.train_step += 1
        return losses

    def save_model(self, root_path, generation):
        torch.save(self.model.state_dict(), root_path + '-model-' + str(generation) + '.pt')

    def load_model(self, torch_path):
        self.model.load_state_dict(torch.load(torch_path))
        print('Pretrained model of ', torch_path, ' has been loaded !!')


# if __name__ == '__main__':
    #
    # start = timeit.default_timer()
    # lr = 0.05
    # batch_size = 32
    # EPOCH = 10
    #
    # train_dataset = FitDataset2D_train()
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    #
    # net = MLP(396)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # loss_func = torch.nn.MSELoss()
    #
    # for epoch in range(EPOCH):
    #     losses = []
    #     for i, data in enumerate(train_loader, 0):
    #         x, y = data
    #         y_hat = net(x)
    #         loss = loss_func(y_hat, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         losses.append(loss.item())
    #
    #     print(y_hat[0].item(), y[0].item())
    #     print('Epoch ', epoch, ' Loss: ',sum(losses) / len(losses), len(losses))
    #
    # # 测试
    # print('Begin evaluation')
    # losses = []
    # dataset_test = FitDataset2D_test()
    # test_loader = DataLoader(dataset=dataset_test, batch_size=10)
    # for i, data in enumerate(test_loader):
    #     x, y = data
    #     y_hat = net(x)
    #     loss = loss_func(y_hat, y)
    #     losses.append(loss)
    # print('The average loss for evaluation is ', (sum(losses) / len(losses)).item())
    # end = timeit.default_timer()
    # print('The overall running time is ', end - start, ' seconds')


# input = 4800
# mlp = MLP(n_feature=input)
# pass
# mlp.initialize_weights()
#
# x = torch.randn([1, input])
# a = timeit.default_timer()
# y = mlp(x)
# b = timeit.default_timer()
# print(y, b-a)
