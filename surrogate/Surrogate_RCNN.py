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
        self.seq_len = 4

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        # return self.x_data[index], self.y_data[index]
        # 根据序列长度返回三维数据
        if index - self.seq_len + 1 < 0:  # 证明需要补0
            return torch.cat(
                [torch.zeros([-(index - self.seq_len + 1), self.x_data[0].shape[0],
                self.x_data[0].shape[1]]), self.x_data[:index + 1]], 0), \
                self.y_data[index]
        else:
            return self.x_data[index - self.seq_len + 1:index + 1], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class Model(nn.Module):
    def __init__(self, hidden_dimension):
        super(Model, self).__init__()  # 继承Net到nn.module

        # 定义层，nn.Linear(in_features,out_features),Linear第一个参数都是层左边输入的特征数，第二个参数是层右边输出的特征数
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 6))
        # self.conv2 = torch.nn.Conv2d(4, 4, (2, 3))
        self.pool1 = torch.nn.MaxPool2d((2, 5))
        # (12-1)//2  *  (N-1)//5 * 4
        # self.pool2 = torch.nn.MaxPool2d((4, 11))

        # self.flatten = torch.nn.Linear(hidden_dimension, 2*hidden_dimension)
        self.rnn = nn.RNN(
            hidden_dimension, 128, 1,
            batch_first=True,
            nonlinearity="relu"
        )

        self.predict = torch.nn.Linear(128, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        with torch.no_grad():
            if type(x) == list:
                x = torch.tensor(x).view(len(x), 1, 12, -1)  # 12 = 5 + 5 + 2 N^T

            xbatch_size = x.shape[0]
            xseq_size = x.shape[1]
        # x = x[:, :, None, :, :]
        x = x.view(-1, x.shape[-2], x.shape[-1])  # 先把 batchsize 和 seq 合并
        x = x[:, None, :, :]
        x1 = self.relu(self.conv1(x))
        x1_p = self.pool1(x1)
        # 先把batch * seqsize 一块给卷了，然后卷完再把batch和seq size给分开
        # x2 = self.relu(self.conv2(x1_p))
        # x2_p = self.pool2(x2)
        x3 = x1_p.view(xbatch_size, xseq_size, x1_p.shape[1], x1_p.shape[2], x1_p.shape[3])
        x3 = x3.view(xbatch_size, xseq_size, -1)
        x4, h_n = self.rnn(x3, None)
        x4 = x4[:, -1, :]
        output = self.predict(x4).view(-1)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


class SurrogateRCNN:
    def __init__(self, buffer_len, units_num):
        N = units_num / 12
        hidden = int(((12-1)//2) * ((N-1)//5) * 4)
        self.model = Model(hidden)  # 代理模型 根据Pooling计算的
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

    def train_model(self, isfirst, istrain=True):  # if is first the epoch will be longer
        train_data = torch.tensor(self.buffer)
        current_dataset = FitDataset2D_train(train_data)
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
