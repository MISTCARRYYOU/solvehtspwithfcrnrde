import torch.nn as nn
import torch.nn.functional as F
from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit


class MLP(nn.Module):

    def __init__(self, n_feature):
        super(MLP, self).__init__()  # 继承Net到nn.module

        self.conv1 = torch.nn.Conv2d(1, 4, (3, 6))
        self.conv2 = torch.nn.Conv2d(4, 4, (2, 3))
        self.conv3 = torch.nn.Conv2d(4, 4, (1, 1))

        self.pool1 = torch.nn.MaxPool2d((2, 5))
        self.pool2 = torch.nn.MaxPool2d((4, 11))

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x[:, None, :, :]
        x1 = self.relu(self.conv1(x))
        x1_p = self.pool1(x1)
        # x2 = self.relu(self.conv2(x1_p))
        # x2_p = self.pool2(x2)
        x3 = self.conv3(x1_p).view(x.shape[0], -1)
        # output = x3.max(dim=1)[0]
        output = torch.mean(x3, dim=1)
        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()


if __name__ == '__main__':

    start = timeit.default_timer()
    lr = 0.05
    batch_size = 32
    EPOCH = 10

    train_dataset = FitDataset2D_train()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    net = MLP(396)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(EPOCH):
        losses = []
        for i, data in enumerate(train_loader, 0):
            x, y = data
            y_hat = net(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(y_hat[0].item(), y[0].item())
        print('Epoch ', epoch, ' Loss: ',sum(losses) / len(losses), len(losses))

    # 测试
    print('Begin evaluation')
    losses = []
    dataset_test = FitDataset2D_test()
    test_loader = DataLoader(dataset=dataset_test, batch_size=10)
    for i, data in enumerate(test_loader):
        x, y = data
        y_hat = net(x)
        loss = loss_func(y_hat, y)
        losses.append(loss)
    print('The average loss for evaluation is ', (sum(losses) / len(losses)).item())
    end = timeit.default_timer()
    print('The overall running time is ', end - start, ' seconds')


