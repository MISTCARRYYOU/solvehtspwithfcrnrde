import torch
import torch.nn as nn
import torch.nn.functional as F
# from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit
from torch.utils.data import Dataset
from collections import deque


# 后续考虑增量式增加数据，而不是all in
class FitDataset1Dseq_train(Dataset):
    def __init__(self, data_tensor):
        self.y_data = torch.log(data_tensor.T[0])
        self.x_data = data_tensor.T[1:].T
        self.len = data_tensor.shape[0]
        self.seq_len = 1  # 序列长度为4

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        # 根据序列长度返回三维数据
        if index - self.seq_len + 1 < 0:  # 证明需要补0
            return torch.cat([torch.zeros([-(index - self.seq_len + 1), self.x_data[0].shape[0]]), self.x_data[:index+1]], 0), \
                self.y_data[index]
        else:
            return self.x_data[index - self.seq_len + 1:index+1], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim):
        """
        input_dim: 每个输入xi的维度
        hidden_dim: 词向量嵌入变换的维度，也就是W的行数
        layer_dim: RNN神经元的层数
        output_dim: 最后线性变换后词向量的维度
        """
        super(Model, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim,
            batch_first=True,
            nonlinearity="relu"
        )

        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        维度说明：
            time_step = sum(像素数) / input_dim
            x : [batch, time_step, input_dim]
        """
        with torch.no_grad():
            if type(x) == list:
                x = torch.tensor(x)

            if len(x.shape) < 3:  # 补充一个seq维度用于评估测试
                assert len(x.shape) == 2
                x = x[:, None, :]

        out, h_n = self.rnn(x, None)  # None表示h0会以全0初始化，及初始记忆量为0
        """
        out : [batch, time_step, hidden_dim]
        """
        out = self.fc1(out[:, -1, :]).view(-1)  # 此处的-1说明我们只取RNN最后输出的那个h。
        return out


class SurrogateRNN:

    def __init__(self, buffer_len, units_num):
        self.model = Model(units_num, 5000, 2)  # 代理模型
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
        current_dataset = FitDataset1Dseq_train(train_data)
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
        print('Training step of {} has epoch losses of {}'.format(self.train_step, losses))
        self.train_step += 1
        return losses

    def save_model(self, root_path, generation):
        torch.save(self.model.state_dict(), root_path + '-model-' + str(generation) + '.pt')

    def load_model(self, torch_path):
        self.model.load_state_dict(torch.load(torch_path))
        print('Pretrained model of ', torch_path, ' has been loaded !!')


# if __name__ == '__main__':
#
#     start = timeit.default_timer()
#     lr = 0.05
#     batch_size = 32
#     EPOCH = 10
#
#     train_dataset = FitDataset1D_train()
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
#
#     net = MLP(396)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     loss_func = torch.nn.MSELoss()
#
#     for epoch in range(EPOCH):
#         losses = []
#         for i, data in enumerate(train_loader, 0):
#             x, y = data
#             y_hat = net(x)
#             loss = loss_func(y_hat, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
#
#         print(y_hat[0].item(), y[0].item())
#         print('Epoch ', epoch, ' Loss: ',sum(losses) / len(losses), len(losses))
#
#     测试
#     print('Begin evaluation')
#     losses = []
#     dataset_test = FitDataset2D_test()
#     test_loader = DataLoader(dataset=dataset_test, batch_size=10)
#     for i, data in enumerate(test_loader):
#         x, y = data
#         y_hat = net(x)
#         loss = loss_func(y_hat, y)
#         losses.append(loss)
#     print('The average loss for evaluation is ', (sum(losses) / len(losses)).item())
#     end = timeit.default_timer()
#     print('The overall running time is ', end - start, ' seconds')


# input = 4800
# mlp = Model(input, 5000, 2, 1)
# pass
#
#
# x = torch.randn([4, input])
# a = timeit.default_timer()
# y = mlp(x)
# b = timeit.default_timer()
# print(y, b-a)
