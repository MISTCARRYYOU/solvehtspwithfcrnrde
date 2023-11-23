from Population import Population
from utilities import *
import random
import math
import copy
import torch
import numpy as np
import timeit

class MultiMet(Population):

    def __init__(self, psize, nn, lb, ub, c_num, e_num, d_num, ce_tnum, m_jnum, m_optnum, evaluate):
        # 初始化一些变量
        super().__init__(psize, nn, lb, ub)  # 初始化父类的一些变量
        self.Cnum = c_num
        self.Enum = e_num
        self.Dnum = d_num
        self.CE_Tnum = ce_tnum
        self.M_Jnum = m_jnum
        self.M_OPTnum = m_optnum

        self.EvaluFunc = evaluate

        # Prob
        self.CETask_Property = [CETask() for _ in range(self.CE_Tnum)]  # 传入的参数不知道做什么？
        self.MTask_Time = [.0 for _ in range(self.M_Jnum*self.M_OPTnum)]
        self.EtoD_Distance = CreateMatrix(self.Enum, self.Dnum)
        self.DtoD_Distance = CreateMatrix(self.Dnum, self.Dnum)
        self.AvailDeviceList = [[] for _ in range(self.M_Jnum * self.M_OPTnum)]
        self.EnergyList = [0. for _ in range(11)]

        self.CloudDevices = [[] for _ in range(self.Cnum)]
        self.EdgeDevices = [[] for _ in range(self.Enum)]
        self.CloudLoad = [[] for _ in range(self.Cnum)]
        self.EdgeLoad = [[] for _ in range(self.Enum)]
        self.DeviceLoad = [[] for _ in range(self.Dnum)]
        self.CETask_coDevice = [[] for _ in range(self.CE_Tnum)]
        self.Edge_Device_comm = [{} for _ in range(self.Enum)]
        self.ST = CreateMatrix(self.M_Jnum, self.M_OPTnum)
        self.ET = CreateMatrix(self.M_Jnum, self.M_OPTnum)
        self.CE_ST = [0. for _ in range(self.CE_Tnum)]
        self.CE_ET = [0. for _ in range(self.CE_Tnum)]

        self.old_and_new = None

        # 定义一些全局变量
        self.TNN = 1000

        self.ibest = CreateMatrix(self.Popsize, self.Nvar)
        self.ibest_fit = [.0 for _ in range(self.Popsize)]

        # PSO
        self.velocity = CreateMatrix(self.Popsize, self.Nvar)
        self.ac1 = .0
        self.ac2 = .0
        self.AC1 = [.0 for _ in range(self.Popsize)]
        self.AC2 = [.0 for _ in range(self.Popsize)]

        self.AW = [.0 for _ in range(self.Popsize)]
        self.OArow = self.Nvar + 1
        self.OA = [[0 for _ in range(self.Nvar)] for __ in range(self.OArow)]

        # ACO
        self.tao_size = 0
        self.ant_tao = CreateMatrix(self.Popsize, self.Nvar + 2)
        self.trial = [0 for _ in range(self.Popsize)]
        self.neigh = [0 for _ in range(self.Popsize)]

        # ABCA
        self.pr = [.0 for _ in range(self.Popsize)]

        # DE
        self.DEstaterecord = [0 for i in range(16)]
        # self.temp_gbest_fit = self.gbest_fit  # 提升效果不明显
        # self.temp_gbest = self.gbest
        self.xs = [random.sample(list(range(0, self.Popsize)), 5) for _ in range(self.Popsize)]  # DE的随机数

    def Initial(self, case):
        task_coDevice_size = [0 for i in range(self.TNN)]
        group_availDev_size = [0 for i in range(self.TNN)]

        # 下面都是读取文件
        t1 = timeit.default_timer()
        fs = file2stream("./data_matrix_"+ str(case)+"00.txt")
        for i in range(0, self.Enum, 1):
            for j in range(0, self.Dnum, 1):
                self.EtoD_Distance[i][j] = fs.pop(0)

        for i in range(0, self.Dnum, 1):
            for j in range(0, self.Dnum, 1):
                self.DtoD_Distance[i][j] = fs.pop(0)

        for i in range(0, self.M_Jnum * self.M_OPTnum, 1):
            self.MTask_Time[i] = fs.pop(0)

        vec_num = 0
        for i in range(0, self.CE_Tnum, 1):

            self.CETask_Property[i].Computation = fs.pop(0)
            self.CETask_Property[i].Communication = fs.pop(0)

            vec_num = fs.pop(0)
            self.CETask_Property[i].Precedence = []
            for j in range(0, vec_num, 1):

                value = fs.pop(0)
                self.CETask_Property[i].Precedence.append(value)

            vec_num = fs.pop(0)
            self.CETask_Property[i].Interact = []
            for j in range(0, vec_num, 1):
                value = fs.pop(0)
                self.CETask_Property[i].Interact.append(value)

            vec_num = fs.pop(0)
            self.CETask_Property[i].Start_Pre = []
            for j in range(0, vec_num, 1):

                value = fs.pop(0)
                self.CETask_Property[i].Start_Pre.append(value)

            vec_num = fs.pop(0)
            self.CETask_Property[i].End_Pre = []
            for j in range(0, vec_num, 1):
                value = fs.pop(0)
                self.CETask_Property[i].End_Pre.append(value)
            self.CETask_Property[i].Job_Constraints = fs.pop(0)

        for i in range(0, self.M_Jnum, 1):
            for j in range(0, self.M_OPTnum, 1):
                vec_num = fs.pop(0)
                self.AvailDeviceList[i * self.M_OPTnum + j] = []
                for k in range(0, vec_num, 1):
                    value = fs.pop(0)
                    self.AvailDeviceList[i * self.M_OPTnum + j].append(value)

        for i in range(0, self.CE_Tnum, 1):
            vec_num = fs.pop(0)
            self.CETask_Property[i].AvailEdgeServerList = []
            for j in range(0, vec_num, 1):
                value = fs.pop(0)
                self.CETask_Property[i].AvailEdgeServerList.append(value)

        for i in range(0, 11, 1):
            self.EnergyList[i] = fs.pop(0)

        assert len(fs) == 0  # 保证文件都读完了，读取文件与C++是一致的
        print('TXT scene data has been loaded !! Time consumes {} seconds'.format(timeit.default_timer() - t1))

        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):
                self.newpop[i][j] = self.pop[i][j] = self.randval(self.Lbound, self.Ubound)

        # PSO ?
        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):
                self.velocity[i][j] = self.randval(self.Lbound, self.Ubound)

        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):
                self.ibest[i][j] = self.pop[i][j]

            # 测试一下目标函数对不对
            # res = self.EvaluFunc(a, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
            #                     self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
            #                        self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
            #                        self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
            #                        self.ET, self.CE_ST, self.CE_ET)
            # pass
            #
            self.pop_fit[i] = self.EvaluFunc(self.pop[i], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                   self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                   self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                   self.ET, self.CE_ST, self.CE_ET)

            # print('pop_fit[i]:', self.pop_fit[i])
            self.newpop_fit[i] = self.pop_fit[i]
            self.ibest_fit[i] = self.pop_fit[i]
        self.worst_and_best()
        for j in range(0, self.Nvar, 1):
            self.gbest[j] = self.pop[self.cur_best][j]
        self.gbest_fit = self.pop_fit[self.cur_best]
        self.CRfit()
        self.CRold = self.CRnew

        # PSO
        self.ac1 = 2
        self.ac2 = 2
        for i in range(0, self.Popsize, 1):
            self.AC1[i] = 2
            self.AC2[i] = 2
            self.AW[i] = 0.85

        self.CreateOA()

        # aco
        for i in range(0, self.Popsize, 1):
            for j in range(0, self.Nvar, 1):
                self.ant_tao[i][j] = self.randval(self.Lbound, self.Ubound)

        for i in range(0, self.Popsize, 1):
            self.ant_tao[i][self.Nvar] = self.EvaluFunc(self.ant_tao[i], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                           self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                           self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                           self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                           self.ET, self.CE_ST, self.CE_ET, isACO=True)
        self.heap_sort(self.ant_tao, self.Popsize, self.Nvar)

        for i in range(0, self.Popsize, 1):
            self.ant_tao[i][self.Nvar + 1] = math.exp(-math.pow(i, 2.0) / (2*math.pow(1e-4*self.Popsize, 2.0))) / (1e-4 * self.Popsize * math.sqrt(2*math.pi))

        for i in range(0, self.Popsize, 1):
            self.trial[i] = 0

        for i in range(0, self.Popsize, 1):
            self.neigh = random.randint(0, self.Nvar) % self.Nvar  # 保证取不到Nvar

        # BA
        self.ne = self.Popsize / 5 * 2
        self.nre = 100
        self.stlim = 10
        self.ngh_decay = 0.8
        self.ngh_origin = (self.Ubound - self.Lbound) * 0.1
        self.ngh = [self.ngh_origin for _ in range(self.Popsize)]
        self.ngh_decay_count = [0 for _ in range(self.Popsize)]

    def GA(self, pc, pm, p_start, p_end):
        self.select(p_start, p_end)
        self.crossover(pc, p_start, p_end)
        self.mutate(pm, p_start, p_end)

    # 选择
    def select(self, p_start, p_end):
        rfitness = [.0 for _ in range(self.Popsize)]
        cfitness = [.0 for _ in range(self.Popsize)]
        sum = 0
        for i in range(0, self.Popsize, 1):
            sum += 1000.0 / self.pop_fit[i]  # 适应值总和
        for i in range(0, self.Popsize, 1):
            rfitness[i] = (1000.0 / self.pop_fit[i]) / sum  # 适应值所占比率
        cfitness[0] = rfitness[0]
        for i in range(1, self.Popsize, 1):
            cfitness[i] = cfitness[i-1] + rfitness[i]           # 轮盘位置

        for i in range(p_start, p_end, 1):
            p = self.randval(0.0, 1.0)
            if p < cfitness[0]:                               # 轮盘赌选择
                for k in range(0, self.Nvar, 1):
                    self.newpop[i][k] = self.pop[p_start][k]
            else:
                for j in range(0, self.Popsize-1, 1):
                    if p >= cfitness[j] and p < cfitness[j+1]:

                        for k in range(0, self.Nvar, 1):
                            self.newpop[i][k] = self.pop[j+1][k]             # 选择的新个体存入newpop

    # 交叉
    def crossover(self, pc, p_start, p_end):
        for mem in range(p_start, p_end, 1):
            while True:
                pos = random.randint(0, self.Popsize-1)
                if pos != mem:
                    break
            p = self.randval(0, 1)
            if p < pc:                          # 若概率小于pc，执行交换子xover操作
                self.xover(pos, mem)

    def xover(self, one, two):
        if (self.Nvar == 2):
            point = 1
        else:
            point = random.randint(0, self.Nvar-1)
        for i in range(0, point, 1):
            r = self.randval(0, 1)
            temp1 = self.newpop[one][i] * r + (1 - r) * self.newpop[two][i]
            temp2 = self.newpop[two][i] * r + (1 - r) * self.newpop[one][i]  # ? 这个xover是单方面的交叉吗？
            if (temp1 > self.Ubound):
                temp1 = self.Ubound
            elif (temp1 < self.Lbound):
                temp1 = self.Lbound
            self.newpop[one][i] = temp1

    # 变异
    def mutate(self, pm, p_start, p_end):
        for i in range(p_start, p_end, 1):
            p = self.randval(0.0, 1.0)
            if p < pm:
                r = random.randint(0, self.Nvar-1)
                self.newpop[i][r] = self.randval(self.Lbound, self.Ubound)

    def Evaluate_gbest(self):  # 评估当代gbest的problems中的具体值
        return self.EvaluFunc(self.gbest, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                   self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                   self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                   self.ET, self.CE_ST, self.CE_ET, is_return_info=True)

    def eval_fitness(self, var, isGT, EAmodel):
        if not isGT:  # 用代理模型评估，这块应该考虑用tensor
            assert type(var[0]) == list  # 此时的var应该是2维的
            fitnesses = torch.exp(EAmodel(var)).tolist()
            print(fitnesses, EAmodel(var).shape, torch.tensor(var).shape)
            print(EAmodel.parameters())
            label = 'SA'
            res = []
            for i in range(len(fitnesses)):
                res.append([label, fitnesses[i], var[i]])
            return res

        else:
            fitness = self.EvaluFunc(var, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                   self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                   self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                   self.ET, self.CE_ST, self.CE_ET)
            label = 'GT'  # ground truth
            return [label, fitness] + var

    # 代理评估策略，使用代理模型和真实评估函数混合评估适应度
    # 真实评估的数据和代理评估的数据都需要被收集！！
    def SAEvaluation(self, s, p_start, p_end, model, generation, collect_generation):
        """
            评估策略：前面全部真实评估加收集数据，后面按比例真实评估；或者不确定性大的再重新用真实评估（MC dropout）
        """
        assert s == 1
        # 前多少代一直用真实评估
        eval_pops = list(range(p_start, p_end, 1))  # 待评估的一共多少个个体
        SA_rate = 1.0  # 后面使用代理模型进行评估的比率
        n_top_reeval = 6  # 前几重新评估
        n_bottom_reeval = 4  # 后几重新评估
        if generation <= collect_generation:
            SA_pop_list = []
        else:
            SA_num = int(SA_rate * len(eval_pops))  # 几个个体进行代理评估
            SA_pop_list = np.random.choice(eval_pops, SA_num, False)

        # 收集评估数据
        eval_reses = []
        GT_pop_list = list(set(eval_pops) - set(SA_pop_list))
        assert len(GT_pop_list) + len(SA_pop_list) == len(eval_pops)
        # 评估GT
        if len(GT_pop_list) != 0:
            for popi in GT_pop_list:
                eval_res = self.eval_fitness(self.newpop[popi], True, None)
                self.newpop_fit[popi] = eval_res[1]
                eval_reses.append(eval_res)
        # 评估SA
        if len(SA_pop_list) != 0:
            newpops = []
            for popi in SA_pop_list:
                newpops.append(self.newpop[popi])
            eval_res = self.eval_fitness(newpops, False, model)  # 按 batch来评估
            # print(eval_res)
            eval_res_fitness = []
            for index_i, popi, in enumerate(SA_pop_list):
                # print(eval_res, len(eval_res))
                self.newpop_fit[popi] = eval_res[index_i][1]
                # print(self.newpop_fit[popi])
                eval_reses.append(eval_res[index_i])
                eval_res_fitness.append(eval_res[index_i][1])
            # 对于SA的评估结果的前n名和后n名用真实函数进行评估
            eval_arg_sort = np.argsort(eval_res_fitness)  # 从小到大
            n_top_reeval = min(n_top_reeval, len(eval_arg_sort))
            n_bottom_reeval = min(n_bottom_reeval, len(eval_arg_sort))
            reeval_pops = [SA_pop_list[eve] for eve in eval_arg_sort[:n_top_reeval]] + \
                        [SA_pop_list[eve] for eve in eval_arg_sort[-n_bottom_reeval:]]
            # print(reeval_pops)
            reeval_pops = list(set(reeval_pops))  # 去除重复元素

            self.old_and_new = []  # 存储新旧两次评估 # popi old new
            # 重新评估reeval_pops中的元素
            for popi in reeval_pops:
                temp = [popi, self.newpop_fit[popi]]
                eval_res = self.eval_fitness(self.newpop[popi], True, None)
                self.newpop_fit[popi] = eval_res[1]
                temp.append(self.newpop_fit[popi])
                eval_reses.append(eval_res)
                self.old_and_new.append(temp)
            with open('./logs/SA_GT_compare.txt', 'a') as f:
                print(self.old_and_new, file=f)

        return eval_reses  # 返回数据

    # 常规评估函数
    def Evaluation(self, s, p_start, p_end):
        if s == 0:
            for i in range(p_start, p_end, 1):
                self.pop_fit[i] = self.EvaluFunc(self.pop[i], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                self.ET, self.CE_ST, self.CE_ET)
        else:
            for i in range(p_start, p_end, 1):
                self.newpop_fit[i] = self.EvaluFunc(self.newpop[i], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                   self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                   self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                   self.ET, self.CE_ST, self.CE_ET)

    def pop_update(self, p_start, p_end):
        for i in range(p_start, p_end, 1):
            for j in range(0, self.Nvar, 1):
                self.pop[i][j] = self.newpop[i][j]
            self.pop_fit[i] = self.newpop_fit[i]
            if self.newpop_fit[i] < self.ibest_fit[i]:
                for j in range(0, self.Nvar, 1):
                    self.ibest[i][j] = self.newpop[i][j]
                self.ibest_fit[i] = self.newpop_fit[i]

    def CreateOA(self):
        u = int(math.log(self.OArow) / math.log(2.0))
        for i in range(0, self.OArow, 1):
            for j in range(0, u, 1):
                b = int(math.pow(2.0, j) - 1)
                tmp = math.floor(i/math.pow(2.0, u-j-1))
                self.OA[i][b] = tmp % 2

        for i in range(0, self.OArow, 1):
            for j in range(0, u, 1):
                b = int(math.pow(2.0, j) - 1)
                for s in range(0, b, 1):
                    self.OA[i][b+s+1] = (self.OA[i][s] + self.OA[i][b]) % 2

    def PSO(self, w, c1, c2, max_ve, p_start, p_end):
        for i in range(p_start, p_end, 1):
            for j in range(0, self.Nvar, 1):
                r1 = self.randval(0, 1)
                r2 = self.randval(0, 1)
                self.velocity[i][j] = w * self.velocity[i][j] + c1 * r1 * (self.ibest[i][j] - self.newpop[i][j]) + c2 * r2 * (self.gbest[j] - self.newpop[i][j])
                if self.velocity[i][j] > max_ve:
                    self.velocity[i][j] = max_ve
                elif self.velocity[i][j] < -max_ve:
                    self.velocity[i][j] = -max_ve
                self.newpop[i][j] = self.newpop[i][j] + self.velocity[i][j]
                if self.newpop[i][j] > self.Ubound:
                    self.newpop[i][j] = self.Lbound
                elif self.newpop[i][j] < self.Lbound:
                    self.newpop[i][j] = self.Ubound

    def ACO(self, epsl, p_start, p_end):
        self.path_finding(epsl, p_start, p_end)
        self.phe_updating(p_start, p_end)

    def path_finding(self, eps1, p_start, p_end):
        l = 0
        psum = 0.0
        rp = [.0 for _ in range(self.Popsize)]
        cp = [.0 for _ in range(self.Popsize)]  # 寻找第l位tao的概率密度和累计概率密度
        ssco = [.0 for _ in range(self.Nvar)]

        for i in range(0, self.Popsize, 1):
            psum += self.ant_tao[i][self.Nvar + 1]

        for i in range(0, self.Popsize, 1):
            rp[i] = self.ant_tao[i][self.Nvar + 1] / psum

        cp[0] = rp[0]
        for i in range(1, self.Popsize, 1):
            cp[i] = cp[i-1] + rp[i]  # 轮盘位置
        for i in range(p_start, p_end, 1):
            p = self.randval(0,1)
            if p < cp[0]:
                l = 0
            else:
                for j in range(0, self.Popsize-1, 1):
                    if p >= cp[j] and p < cp[j+1]:
                        l = j + 1
                        break
            for j in range(0, self.Nvar, 1):
                for k in range(0, self.Popsize, 1):
                    ssco[j] += abs(self.ant_tao[k][j] - self.ant_tao[l][j]) / (self.Popsize - 1.0)
                ssco[j] *= eps1

            for j in range(0, self.Nvar, 1):
                if self.randval(0, 1) < 0.15:
                    self.newpop[i][j] = self.ant_tao[l][j] + (self.gauss()*math.sqrt(ssco[j]))
                    if self.newpop[i][j] < self.Lbound:
                        self.newpop[i][j] = self.Lbound
                    elif self.newpop[i][j] > self.Ubound:
                        self.newpop[i][j] = self.Ubound
                else:
                    self.newpop[i][j] = self.pop[i][j]

    def phe_updating(self, p_start, p_end):

        for i in range(p_start, p_end, 1):
            sameflag = False
            for j in range(0, self.Popsize, 1):
                if self.ant_tao[j][self.Nvar] == self.newpop_fit[i]:
                    sameflag = True
            if sameflag == False:
                maxTao = -1e5
                maxIndex = 0
                for j in range(0, self.Popsize, 1):
                    if self.ant_tao[j][self.Nvar] > maxTao:
                        maxTao = self.ant_tao[j][self.Nvar]
                        maxIndex = j
                if self.ibest_fit[i] < maxTao:
                    for j in range(0, self.Nvar, 1):
                        self.ant_tao[maxIndex][j] = self.newpop[i][j]
                    self.ant_tao[maxIndex][self.Nvar] = self.newpop_fit[i]
        self.heap_sort(self.ant_tao, self.Popsize, self.Nvar)
        for i in range(0, self.Popsize, 1):
            self.ant_tao[i][self.Nvar + 1] = math.exp(-math.pow(i, 2.0) / (2 * math.pow(1e-4 * self.Popsize, 2.0))) / (1e-4 * self.Popsize * math.sqrt(2 * math.pi))

    def BA(self, p_start, p_end, psize):
        self.pop_heap_sort(psize)
        self.newpop_heap_sort(psize)
        for i in range(0, int(self.ne), 1):
            self.NeighborFlowerPatch(self.nre, i)
        if self.randval(0, 1) < 0.5:
            self.DE(self.randval(0.1, 0.9), random.randint(0, 4), self.randval(0.1, 0.9), int(self.ne), psize)
        else:
            self.mutate(1, int(self.ne), psize)

    def pop_heap_sort(self, psize):
        for i in range(int(psize/2 -1), -1, -1):
            self.pop_heap_adjust(i, psize)

        for i in range(psize-1, 0, -1):
            temp = self.pop[0]
            temp_fit = self.pop_fit[0]
            p_ngh = self.ngh[0]
            self.pop[0] = self.pop[i]
            self.pop_fit[0] = self.pop_fit[i]
            self.ngh[0] = self.ngh[i]
            self.ngh[i] = p_ngh
            self.pop[i] = temp
            self.pop_fit[i] = temp_fit
            self.pop_heap_adjust(0, i)

    def pop_heap_adjust(self, s, length):
        temp = self.pop[s]
        temp_fit = self.pop_fit[s]
        i = 2*s + 1
        while i < length:
            if i < (length - 1) and self.pop_fit[i] < self.pop_fit[i + 1]:
                i += 1
            if temp_fit > self.pop_fit[i]:
                break
            self.pop[s] = self.pop[i]
            self.pop_fit[s] = self.pop_fit[i]
            s = i
            i = 2*s + 1

        self.pop[s] = temp
        self.pop_fit[s] = temp_fit

    def newpop_heap_sort(self, length):
        for i in range(int(length/2 -1), -1, -1):
            self.newpop_heap_adjust(i, length)
        for i in range(length-1, 0, -1):
            temp = self.newpop[0]
            temp_fit = self.newpop_fit[0]
            self.newpop[0] = self.newpop[i]
            self.newpop_fit[0] = self.newpop_fit[i]
            self.newpop[i] = temp
            self.newpop_fit[i] = temp_fit
            self.newpop_heap_adjust(0, i)

    def newpop_heap_adjust(self, s, length):
        temp = self.newpop[s]
        temp_fit = self.newpop_fit[s]
        i = 2*s + 1
        while i < length:
            if i < (length - 1) and self.newpop_fit[i] < self.newpop_fit[i + 1]:
                i += 1
            if temp_fit > self.newpop_fit[i]:
                break
            self.newpop[s] = self.newpop[i]
            self.newpop_fit[s] = self.newpop_fit[i]
            s = i
            i = 2*i + 1
        self.newpop[s] = temp
        self.newpop_fit[s] = temp_fit

    def NeighborFlowerPatch(self, nr, point):

        self.newpop_bit_climbing(point, self.ngh[point], nr)
        if self.pop_fit[point] < self.newpop_fit[point]:
            self.ngh[point] *= self.ngh_decay
            self.ngh_decay_count[point] += 1
            if self.ngh_decay_count[point] > self.stlim:
                for j in range(0, self.Nvar, 1):
                    self.newpop[point][j] = self.randval(self.Lbound, self.Ubound)
                self.newpop_fit[point] = self.EvaluFunc(self.newpop[point], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                self.ET, self.CE_ST, self.CE_ET)
                self.ngh[point] = self.ngh_origin
                self.ngh_decay_count[point] = 0

    def newpop_bit_climbing(self, popi, L, scale):
        temp = [.0 for _ in range(self.Nvar)]
        permu = [i for i in range(self.Nvar)]
        random.shuffle(permu)  # 随机打乱
        for j in range(0, int(L), 1):
            for k in range(0, self.Nvar, 1):
                temp[k] = self.newpop[popi][k]
            bit = permu[j % self.Nvar]
            temp[bit] += scale * self.randval(self.Lbound, self.Ubound)
            temp_fit = self.EvaluFunc(temp, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                                self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                                self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                                self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                                self.ET, self.CE_ST, self.CE_ET)
            if temp_fit < self.newpop_fit[popi]:
                self.newpop[popi][bit] = temp[bit]
                self.newpop_fit[popi] = temp_fit

    def DE(self, F, S, cr, p_start, p_end, isRL=False):
        self.differential_mutate(F, S, p_start, p_end, isRL=isRL)
        self.differential_crossover(cr, p_start, p_end)

    # 一般是全部DE被调用后更新
    def update_xs(self):
        self.xs = [random.sample(list(range(0, self.Popsize)), 5) for _ in range(self.Popsize)]  # DE的随机数

    def differential_mutate(self, F, S, p_start, p_end, isRL):
        if isRL:
            assert p_end - p_start == 1
            x = self.xs[p_start]
        else:  # 否则是正常生成随机数
            x = random.sample(list(range(0, self.Popsize)), 5)

        for i in range(p_start, p_end, 1):
            assert len(list(set(x))) == len(x)
            for j in range(0, self.Nvar, 1):
                if S == 1:
                    self.newpop[i][j] = self.gbest[j] + F * (self.ibest[x[0]][j] - self.ibest[x[1]][j])
                elif S == 2:
                    self.newpop[i][j] = self.ibest[x[0]][j] + F * (self.ibest[x[1]][j] - self.ibest[x[2]][j]) + F * (self.ibest[x[3]][j] - self.ibest[x[4]][j])
                elif S == 3:
                    self.newpop[i][j] = self.gbest[j] + F * (self.ibest[x[0]][j] - self.ibest[x[1]][j]) + F * (self.ibest[x[2]][j] - self.ibest[x[3]][j])
                elif S == 4:
                    self.newpop[i][j] = self.ibest[i][j] + F * (self.gbest[j] - self.ibest[i][j]) + F * (self.ibest[x[0]][j] - self.ibest[x[1]][j])
                elif S == 0:
                    self.newpop[i][j] = self.ibest[x[0]][j] + F * (self.ibest[x[1]][j] - self.ibest[x[2]][j])
                # bounding
                if (self.newpop[i][j] > self.Ubound):
                    self.newpop[i][j] = self.Ubound
                elif (self.newpop[i][j] < self.Lbound):
                    self.newpop[i][j] = self.Lbound

    def differential_crossover(self, cr, p_start, p_end):
        for i in range(p_start, p_end, 1):
            d = random.randint(0, self.Nvar)
            for j in range(0, self.Nvar, 1):
                r = self.randval(0, 1)
                if r > cr and j != d:
                    self.newpop[i][j] = self.pop[i][j]

    def ABCA(self, limit, p_start, p_end):
        tmp = [.0 for _ in range(self.Nvar)]
        tmp_fit = 0
        for i in range(0, self.Popsize, 1):
            self.EmployedBee(i, tmp, self.pop)
            tmp_fit = self.EvaluFunc(tmp, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                            self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                            self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                            self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                            self.ET, self.CE_ST, self.CE_ET)
            if (tmp_fit < self.pop_fit[i]):
                for j in range(0, self.Nvar, 1):
                    self.newpop[i][j] = tmp[j]
                self.newpop_fit[i] = tmp_fit
            else:
                for j in range(0, self.Nvar, 1):
                    self.newpop[i][j] = self.pop[i][j]
                self.newpop_fit[i] = self.pop_fit[i]
                self.trial[i] += 1

        self.OnlookerBee(0, self.Popsize)
        T = 0
        i = p_start
        totaliter = 0
        while (T < (p_end - p_start) and totaliter < 2 * self.Popsize):

            r = self.randval(0, 1)
            if (r < self.pr[i]):
                T += 1
                self.EmployedBee(i, tmp, self.newpop)
                tmp_fit = self.EvaluFunc(tmp, self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                            self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                            self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                            self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                            self.ET, self.CE_ST, self.CE_ET)
                if (tmp_fit < self.newpop_fit[i]):
                    for j in range(0, self.Nvar, 1):
                        self.newpop[i][j] = tmp[j]
                    self.newpop_fit[i] = tmp_fit
                else:
                    self.trial[i] += 1
            i += 1
            if (i >= p_end):
                i = p_start
            totaliter += 1
        self.ScoutBee(limit, p_start, p_end)

    def EmployedBee(self, pn, tmp, pp):
        for i in range(0, self.Nvar, 1):
            tmp[i] = self.pop[pn][i]
        para2change = random.randint(0, self.Nvar-1)
        while True:
            neighbor = random.randint(0, self.Popsize-1)
            if neighbor != pn:
                break
        tmp[para2change] = tmp[para2change] + (pp[neighbor][para2change] - tmp[para2change])* self.randval(-1, 1)
        if (tmp[para2change] < self.Lbound):
            tmp[para2change] = self.Lbound
        if (tmp[para2change] > self.Ubound):
            tmp[para2change] = self.Ubound

    def OnlookerBee(self, p_start, p_end):
        maxf = 0
        for i in range(p_start, p_end, 1):
            self.pr[i] = math.exp(self.newpop_fit[i] / 1000)
            if self.pr[i] > 1e10:
                self.pr[i] = 1e10
            if self.pr[i] > maxf:
                maxf = self.pr[i]
        for i in range(p_start, p_end, 1):
            if maxf != 0:
                self.pr[i] = 0.9 * self.pr[i] / maxf + 0.1
            else:
                self.pr[i] = 1

    def ScoutBee(self, limit, p_start, p_end):
        maxindex = p_start
        for i in range(p_start + 1, p_end, 1):
            if self.trial[i] > self.trial[maxindex]:
                maxindex = i
        if self.trial[maxindex] > limit:
            for i in range(0, self.Nvar, 1):
                self.newpop[maxindex][i] = self.randval(self.Lbound, self.Ubound)
            self.newpop_fit[maxindex] = self.EvaluFunc(self.newpop[maxindex], self.Cnum, self.Enum, self.Dnum, self.CE_Tnum, self.M_Jnum, self.M_OPTnum,
                            self.CETask_Property, self.MTask_Time, self.EtoD_Distance, self.DtoD_Distance,
                            self.AvailDeviceList, self.EnergyList, self.CloudDevices, self.EdgeDevices, self.CloudLoad,
                            self.EdgeLoad, self.DeviceLoad, self.CETask_coDevice, self.Edge_Device_comm, self.ST,
                            self.ET, self.CE_ST, self.CE_ET)
