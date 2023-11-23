import numpy as np
# 云边任务结构体
class CETask:
    def __init__(self):
        self.Computation = 0.
        self.Communication = 0.
        self.Precedence = []
        self.Interact = []
        self.Start_Pre = []
        self.End_Pre = []
        self.Job_Constraints = 0
        self.AvailEdgeServerList = []


# 生成二维行列矩阵
def CreateMatrix(row, col):
    return np.zeros([row, col]).tolist()


def file2stream(path):
    res = []
    with open(path, 'r') as f:
        for eve in f:
            res += [float(each) if '.' in each else int(each) for each in eve.strip('\n').split()]
    return res


# 计算两个个体间的欧式距离
def OE_distance(pop1, pop2):
    p1 = np.array(pop1)
    p2 = np.array(pop2)
    return np.sqrt(np.sum((p1 -p2)**2))


# 将十进制转化为二进制，pop size
def onehot_coding(num, pop_size):
    bits = len(str(bin(pop_size))) - 2
    tmp = str(bin(num))
    addedzero = bits + 2 - len(tmp)
    res = '0'*addedzero + tmp[2:]
    res2 = []
    for eve in res:
        res2.append(int(eve))
    return res2
