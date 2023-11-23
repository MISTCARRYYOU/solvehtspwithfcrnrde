'''
将采集的10000条样本按照7:3划分训练集和测试集，并打乱顺序
'''
import random

def txt2list(_path):
    with open(_path, 'r') as f:
        res = [eve for eve in f]
    return res


paths = [
    'buffer-2022-09-19 17-15-46.txt',
    'buffer-2022-09-19 17-22-17.txt'
]

reses = []
for path in paths:
    reses += txt2list(path)
print(len(reses))
random.shuffle(reses)
print(len(reses))

with open('./buffer-10000-GA-DE.txt', 'w') as f:
    for eve in reses:
        print(eve, file=f, end='')
