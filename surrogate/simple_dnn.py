import torch.nn as nn
import torch.nn.functional as F
from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit


class MLP(nn.Module):
    """
    这是一个最简单的一层的拟合的神经网络。hidden是隐藏层，predict是预测层
    """
    def __init__(self, n_feature):
        super(MLP, self).__init__()  # 继承Net到nn.module

        self.hidden = nn.Linear(n_feature, 1024)
        self.hidden2 = nn.Linear(1024, 256)
        self.predict = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))  # x经过一个隐藏层，然后再被一个relu函数激活一下
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


if __name__ == '__main__':

    start = timeit.default_timer()

    # 训练
    lr = 0.05
    batch_size = 32
    EPOCH = 10

    dataset_train = FitDataset_train()
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size)

    net = MLP(4800)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(EPOCH):
        losses = []
        for i, data in enumerate(train_loader):
            x, y = data
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(y_hat[0].item(), y[0].item())
        print('Epoch ', epoch, ' Loss: ', sum(losses) / len(losses), len(losses))

    # 测试
    print('Begin evaluation')
    losses = []
    dataset_test = FitDataset_test()
    test_loader = DataLoader(dataset=dataset_test, batch_size=10)
    for i, data in enumerate(test_loader):
        x, y = data
        y_hat = net(x)
        loss = loss_func(y_hat, y)
        losses.append(loss)
    print('The average loss for evaluation is ', (sum(losses) / len(losses)).item())
    end = timeit.default_timer()
    print('The overall running time is ', end - start, ' seconds')

