import torch.nn as nn
import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from surrogate.Condigure import *
from torch.utils.data import DataLoader
import timeit
from torch.utils.data import Dataset
from collections import deque

# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)
# print(out.shape)

# 后续考虑增量式增加数据，而不是all in
class FitDataset1Dseq_train(Dataset):
    def __init__(self, data_tensor):
        self.y_data = torch.log(data_tensor.T[0])
        self.x_data = data_tensor.T[1:].T
        self.len = data_tensor.shape[0]
        self.seq_len = 4  # 序列长度为4

    def __getitem__(self, index):  # 支持下标操作，根据索引获取数据
        # 根据序列长度返回三维数据
        if index - self.seq_len + 1 < 0:  # 证明需要补0
            return torch.cat([torch.zeros([-(index - self.seq_len + 1), self.x_data[0].shape[0]]), self.x_data[:index+1]], 0), \
                self.y_data[index]
        else:
            return self.x_data[index - self.seq_len + 1:index+1], self.y_data[index]

        # return self.x_data[index], self.y_data[index]

    def __len__(self):  # 获取数据条数
        return self.len


#### positional encoding ####
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Model(nn.Module):
    def __init__(self, feature_size, num_layers=1, dropout=0):  # feature_size 表示特征维度（必须是head的整数倍）, num_layers 表示 Encoder_layer 的层数， dropout 用于防止过你和
        super(Model, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.hidden_size = 512
        self.embad = nn.Linear(feature_size, self.hidden_size)
        self.pos_encoder = PositionalEncoding(self.hidden_size)  #位置编码前要做归一化，否则捕获不到位置信息
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.hidden_size, 1)  # 这里用全连接层代替了decoder
        self.init_weights()
        # self.src_key_padding_mask = None  # 后面用了掩码~

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # if self.src_key_padding_mask is None:
        #     mask_key = src_padding.bool()
        #     self.src_key_padding_mask = mask_key
        # batch seq hidden
        with torch.no_grad():
            if type(src) == list:
                src = torch.tensor(src)
            if len(src.shape) == 2:  # 补充一个seq用于评估测试
                src = src[:, None, :]
            bsize = src.shape[0]
            ssize = src.shape[1]
        src = src.view(-1, src.shape[-1])
        src = self.embad(src)
        src = src.view(bsize, ssize, -1)
        src = src.transpose(0, 1)  # 调换seq & batch的位置
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # 注意transformer接受的是seq x batch x hidden
        output = self.decoder(output)[-1, :, :].view(-1)
        return output


class SurrogateTransformer:

    def __init__(self, buffer_len, unit_num):
        self.model = Model(unit_num)  # 代理模型
        self.buffer = deque(maxlen=buffer_len)  # 存放数据的，格式为[fitness, var]
        # ------------------------- 训练参数 ---------------------------
        self.lr = 0.0001
        self.batch_size = 16
        self.EPOCH_first = 10
        self.EPOCH_increm = 2
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
