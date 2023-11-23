import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit


class MLP(nn.Module):
    """
    这是一个最简单的一层的拟合的神经网络。hidden是隐藏层，predict是预测层
    """
    def __init__(self, n_feature,  n_output=1):
        super(MLP, self).__init__()  # 继承Net到nn.module

        # 定义层，nn.Linear(in_features,out_features),Linear第一个参数都是层左边输入的特征数，第二个参数是层右边输出的特征数
        self.hidden = nn.Linear(n_feature, 5000)  # 一层隐藏层,n_feature个特征，n_hidden个神经元
        self.hidden2 = nn.Linear(5000, 512)
        # self.hidden3 = nn.Linear(1024, 512)
        self.predict = nn.Linear(512, n_output)  # 预测层,n_hidden个神经元，n_output个特征的输出

    def forward(self, x):
        x = F.relu(self.hidden(x))  # x经过一个隐藏层，然后再被一个relu函数激活一下
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 1, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

#
input = 12000
mlp = MLP(n_feature=input)
pass
# mlp.initialize_weights()

x = torch.randn([1, input])
a = timeit.default_timer()
y = mlp(x)
b = timeit.default_timer()
print(y, b-a)
