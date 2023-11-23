from utilities import *
import random
import math


class Population:
    def __init__(self, psize, nn, lb, ub):
        self.Nvar = nn
        self.Popsize = psize
        self.Lbound = lb
        self.Ubound = ub
        self.stored = False
        self.pop_fit = 0
        self.newpop_fit = 0
        self.gbest_fit = 0
        self.cur_best = 0
        self.cur_worst = 0
        self.CRold = 0
        self.CRnew = 0

        self.pop = CreateMatrix(self.Popsize, self.Nvar)
        self.newpop = CreateMatrix(self.Popsize, self.Nvar)
        self.gbest = [None for _ in range(self.Nvar)]
        self.pop_fit = [.0 for _ in range(self.Popsize)]
        self.newpop_fit = [.0 for _ in range(self.Popsize)]

        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):

                self.pop[i][j] = self.randval(self.Lbound, self.Ubound)

        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):
                self.newpop[i][j] = self.randval(self.Lbound, self.Ubound)

        for i in range(0, self.Nvar, 1):
            self.gbest[i] = self.randval(self.Lbound, self.Ubound)

    def randval(self, low, high):
        return (random.random()) * (high - low) + low

    def worst_and_best(self):
        self.cur_best = 0
        self.cur_worst = 0
        for i in range(0, self.Popsize, 1):
            if self.pop_fit[i] < self.pop_fit[self.cur_best]:
                self.cur_best = i
            elif self.pop_fit[i] > self.pop_fit[self.cur_worst]:
                self.cur_worst = i

    # 这里的worst and best 需要进行一些调整
    def SA_worst_and_best(self):
        self.cur_best = 0
        self.cur_worst = 0
        for i in range(0, self.Popsize, 1):
            if self.pop_fit[i] < self.pop_fit[self.cur_best]:
                self.cur_best = i
            elif self.pop_fit[i] > self.pop_fit[self.cur_worst]:
                self.cur_worst = i
        # 确认一下最好的是不是GT评估的，因为预测的前n个最好的都GT reeval了，所以其实已经有容错了
        if self.old_and_new is not None:  # 前期不进行代理评估
            GT_popi = [eve[0] for eve in self.old_and_new]
            GT_true = [eve[1] for eve in self.old_and_new]
            if self.cur_best not in GT_popi:  # 证明不一致
                GT_cur_best = GT_popi[np.argsort(GT_true)[0]]
                assert GT_cur_best != self.cur_best  # 一定是满足的
                self.cur_best = GT_cur_best

    # 这个函数保证了每次要不然就是最差的变成当前最好的，要不然就是最好的变成当前最好的
    def Elist(self):
        if self.pop_fit[self.cur_best] < self.gbest_fit:
            for i in range(0, self.Nvar, 1):
                self.gbest[i] = self.pop[self.cur_best][i]
            self.gbest_fit = self.pop_fit[self.cur_best]
        else:
            for i in range(0, self.Nvar, 1):
                self.pop[self.cur_worst][i] = self.gbest[i]
            self.pop_fit[self.cur_worst] = self.gbest_fit

    def CRfit(self):
        self.CRold = self.CRnew
        ave = self.average_fit() / self.pop_fit[self.cur_worst]
        best = self.pop_fit[self.cur_best] / self.pop_fit[self.cur_worst]
        if best == 1:
            self.CRnew = 1
        else:
            self.CRnew = (1 - ave) / (1 - best)

    def average_fit(self):
        ave = 0
        for i in range(0, self.Popsize, 1):
            ave += self.pop_fit[i]
        ave /= self.Popsize
        return ave

    def heap_sort(self, num, length, cbit):
        for i in range(int(length/2 -1), -1, -1):
            self.heap_adjust(num, i, length, cbit)
        for i in range(length-1, 0, -1):
            temp = num[0]
            num[0] = num[i]
            num[i] = temp
            self.heap_adjust(num, 0, i, cbit)

    def heap_adjust(self, num, s, length, cbit):
        temp = num[s]  # 这是个一维数组
        i = 2*s + 1
        while i < length:
            if i < (length - 1) and num[i][cbit] < num[i+1][cbit]:
                i += 1
            if temp[cbit] > num[i][cbit]:
                break
            num[s] = num[i]
            s = i
            i = 2*i + 1
        num[s] = temp

    def gauss(self):
        return np.random.normal(loc=0.0, scale=1.0)
