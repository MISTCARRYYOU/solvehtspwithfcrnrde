import math
import random
QN = 23
import timeit
from surrogate.Condigure import *


def heap_adjust(num, s, length, cbit):
    temp = num[s]
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


def randnorm(miu, score):
    return miu + score * math.sqrt(-2 * math.log(random.random())) * math.cos(2 * math.pi * random.random())


def heap_sort(num, length, cbit):
    i = int(length / 2) - 1
    while i >= 0:
        heap_adjust(num, i, length, cbit)
        i -= 1

    i = length - 1
    while i > 0:
        temp = num[0]
        num[0] = num[i]
        num[i] = temp
        heap_adjust(num, 0, i, cbit)
        i -= 1


# 目标函数
def CED_Schedule(var, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
                 MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList,
                 EnergyList, CloudDevices, EdgeDevices, CloudLoad, EdgeLoad,
                 DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET, isACO=False, is_return_info=False):
    obj_start = timeit.default_timer()
    ce_sele = [0 for _ in range(CE_Tnum)]
    cevar = [0 for _ in range(CE_Tnum)]
    mvar = [0 for _ in range(M_Jnum * M_OPTnum)]
    seq_var = [0 for _ in range(M_Jnum * M_OPTnum)]
    sort_vec = [[0., 0.] for _ in range(M_Jnum * M_OPTnum)]

    for i in range(0, CE_Tnum, 1):
        if var[i] > 0.5:
            ce_sele[i] = 1
        else:
            ce_sele[i] = 0
        if ce_sele[i] == 0:
            cevar[i] = int(var[CE_Tnum + i] * (Cnum - 1))
        else:
            cevar[i] = int(var[CE_Tnum + i]*(len(CETask_Property[i].AvailEdgeServerList) - 1))
            cevar[i] = CETask_Property[i].AvailEdgeServerList[cevar[i]]

    # argsort操作
    for i in range(0, M_Jnum * M_OPTnum, 1):
        sort_vec[i][0] = var[2 * CE_Tnum + i]
        sort_vec[i][1] = i
    heap_sort(sort_vec, M_Jnum * M_OPTnum, 0)  # 把sort_vec中的元素进行排序

    for i in range(0, M_Jnum * M_OPTnum, 1):
        seq_var[i] = sort_vec[i][1]

    for i in range(0, M_Jnum * M_OPTnum, 1):
        mvar[i] = int(var[CE_Tnum * 2 + M_Jnum * M_OPTnum + i] * (len(AvailDeviceList[seq_var[i]]) - 1))
        mvar[i] = AvailDeviceList[seq_var[i]][mvar[i]]  # 已经分好了要去哪个设备上了

    CloudDevices = [[] for _ in range(Cnum)]
    EdgeDevices = [[] for _ in range(Enum)]
    CloudLoad = [[] for _ in range(Cnum)]
    EdgeLoad = [[] for _ in range(Enum)]
    DeviceLoad = [[] for _ in range(Dnum)]
    CETask_coDevice = [[] for _ in range(CE_Tnum)]
    Edge_Device_comm = [{} for _ in range(Enum)]

    for i in range(0, M_Jnum, 1):
        for j in range(0, M_OPTnum, 1):
            ST[i][j] = 0  # start time
            ET[i][j] = 0  # end time

    geneO = [-1 for _ in range(M_Jnum)]

    # print('mvar', mvar)
    for i in range(0, M_Jnum * M_OPTnum, 1):
        CJ = int(seq_var[i] / M_OPTnum)
        geneO[CJ] += 1
        CO = geneO[CJ]
        CM = mvar[seq_var[i]]
        Cprev = CO - 1
        if Cprev < 0:  # 代表刚开始加工
            if len(DeviceLoad[CM]) == 0:  # target machine has no operation
                ST[CJ][CO] = 0
                ET[CJ][CO] = MTask_Time[CJ * M_OPTnum + CO]
                DeviceLoad[CM].append(CJ * M_OPTnum + CO)
            else:
                Qprev = DeviceLoad[CM][len(DeviceLoad[CM]) - 1]  # 最后一个元素作为Qprev
                ST[CJ][CO] = ET[int(Qprev / M_OPTnum)][Qprev % M_OPTnum]
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO]
                DeviceLoad[CM].append(CJ * M_OPTnum + CO)
        else:
            if len(DeviceLoad[CM]) == 0:
                ST[CJ][CO] = ET[CJ][Cprev]
                # 制造任务考虑物流相当于
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] + DtoD_Distance[mvar[CJ * M_OPTnum + Cprev]][mvar[CJ * M_OPTnum + CO]] / (100000.0 / 3600.0)
                DeviceLoad[CM].append(CJ * M_OPTnum + CO)
            else:  # 既有前驱工序，又得等位
                _iter = len(DeviceLoad[CM])  # 这里原来的.end()返回最后一步的指针
                _iter -= 1
                Qprev = DeviceLoad[CM][_iter]
                ST[CJ][CO] = max(ET[CJ][Cprev], ET[int(Qprev / M_OPTnum)][Qprev % M_OPTnum])
                ET[CJ][CO] = ST[CJ][CO] + MTask_Time[CJ * M_OPTnum + CO] + DtoD_Distance[mvar[CJ * M_OPTnum + Cprev]][mvar[CJ * M_OPTnum + CO]] / (100000.0 / 3600.0)
                DeviceLoad[CM].append(CJ * M_OPTnum + CO)
    # 下面的操作就是让每5个机器为一组，放到云边协同设备里
    for i in range(0, CE_Tnum, 1):
        for j in range(0, M_OPTnum, 1):
            # duplicate = find(CETask_coDevice[i].begin(), CETask_coDevice[i].end(), mvar[i * M_OPTnum + j])
            duplicate = mvar[i * M_OPTnum + j] in CETask_coDevice[i]
            if duplicate == False:
                CETask_coDevice[i].append(mvar[i * M_OPTnum + j])
    # 下面开始处理计算任务， 把机器device对应到了云端或者边缘端
    for i in range(0, CE_Tnum, 1):
        theJob = i
        if (ce_sele[i] == 0):  # cloud mode
            for j in range(0, M_OPTnum, 1):
                duplicate = mvar[theJob * M_OPTnum + j] in CloudDevices[cevar[i]]
                if duplicate == False:
                    CloudDevices[cevar[i]].append(mvar[theJob * M_OPTnum + j])
            CloudLoad[cevar[i]].append(i)
        else:
            for j in range(0, M_OPTnum, 1):
                duplicate = mvar[theJob * M_OPTnum + j] in EdgeDevices[cevar[i]]
                if duplicate == False:
                    EdgeDevices[cevar[i]].append(mvar[theJob * M_OPTnum + j])
            EdgeLoad[cevar[i]].append(i)

    nearest_device = [0 for _ in range(Enum)]  # 计算边缘端到设备端的距离
    for i in range(0, Enum, 1):
        min_dis = 0
        min_index = 0
        for j in range(0, Dnum, 1):
            if min_dis > EtoD_Distance[i][j]:
                min_dis = EtoD_Distance[i][j]
                min_index = j
        nearest_device[i] = min_index

    for i in range(0, Enum, 1):
        EdgeDevices[i].append(nearest_device[i])  # each device connects the nearest edge for data forwarding.

    nearest_edge = [0 for _ in range(Dnum)]

    for i in range(0, Dnum, 1):
        min_dis = 1e10
        min_index = 0
        for j in range(0, Enum, 1):
            if min_dis > EtoD_Distance[j][i]:
                min_dis = EtoD_Distance[j][i]
                min_index = j
        nearest_edge[i] = min_index

    Edge_smallest_rate = [.0 for _ in range(Enum)]
    # 这一段就是计算边缘到设备的通讯相关内容
    for i in range(0, Enum, 1):  # 这里应该是计算通讯相关的内容 边缘段到设备段的通讯
        Edge_smallest_rate[i] = 1e10
        bottom_sum = 0.000  # python这里没有double类型的，只有float类型，与c++比会有累计误差
        for star_iter in EdgeDevices[i]:
            bottom_sum += QN / pow(EtoD_Distance[i][star_iter] / 1000, 1.0)  # 因为会分带宽 所以先全部计算
        for star_iter in EdgeDevices[i]:
            current_gain = QN / pow(EtoD_Distance[i][star_iter] / 1000, 1.0)
            transmission_rate = 20 * math.log2(1 + current_gain / abs(bottom_sum - current_gain - 100)) / 8.0 # Mbps -> MB / s
            Edge_Device_comm[i][star_iter] = transmission_rate
            if transmission_rate < Edge_smallest_rate[i]:
                Edge_smallest_rate[i] = transmission_rate
    for i in range(0, CE_Tnum, 1):
        CE_ST[i] = CE_ET[i] = 0
    time_max = 0
    energy = 0
    n1, n2, n3, n4 =0, 0, 0, 0
    for i in range(0, CE_Tnum, 1):  # 这部分计算的是计算任务与制造任务间的通讯时间和能量消耗
        t_comm = 0
        t_comp = 0
        if ce_sele[i] == 0:  # cloud mode
            for star_iter in CETask_coDevice[i]:  # CETask_coDevice[i] 装的是计算任务i对应的制造设备编号  # ？ 云和边缘的通讯忽略不计了吗？
                cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_smallest_rate[nearest_edge[star_iter]])  # 为什么设备与云的通讯，可以转为设备最近边的最小率
                energy += cur_comm * QN / 1000  # offloading energy qn * bn / rn(a)
                if cur_comm > t_comm:
                    t_comm = cur_comm
        else:
            for star_iter in CETask_coDevice[i]:
                cur_comm = CETask_Property[i].Communication * 10 / (1000 * Edge_Device_comm[cevar[i]][star_iter])
                energy += cur_comm * QN / 1000  # offloading energy qn * bn / rn(a)
                if cur_comm > t_comm:
                    t_comm = cur_comm
        if ce_sele[i] == False:  # cloud mode
            t_comp = CETask_Property[i].Computation / 3.7
        else:  # edge mode
            if len(EdgeLoad[cevar[i]]) < 6:
                t_comp = CETask_Property[i].Computation / 2.2
            else:
                t_comp = CETask_Property[i].Computation / 2.2 * len(EdgeLoad[cevar[i]])
        theJob = i
        # 统计四类制造-计算关系的数量
        if CETask_Property[i].Job_Constraints == 0:
            n1 += 1
        elif CETask_Property[i].Job_Constraints == 1:
            n2 += 1
        elif CETask_Property[i].Job_Constraints == 2:
            n3 += 1
        else:
            n4 += 1
        # 制造-计算开始顺序
        if CETask_Property[i].Job_Constraints == 0 or CETask_Property[i].Job_Constraints == 2:
            CE_ST[i] = min(CE_ST[i], ST[theJob][M_OPTnum - 1])
        elif CETask_Property[i].Job_Constraints == 1 or CETask_Property[i].Job_Constraints == 3:
            CE_ST[i] = max(CE_ST[i], ST[theJob][M_OPTnum - 1])
        CE_ET[i] = CE_ST[i] + t_comm + t_comp
        # 制造-计算结束顺序
        if CETask_Property[i].Job_Constraints == 1 or CETask_Property[i].Job_Constraints == 2:
            CE_ET[i] = min(CE_ET[i], ET[theJob][M_OPTnum - 1])
        elif CETask_Property[i].Job_Constraints == 0 or CETask_Property[i].Job_Constraints == 3:
            CE_ET[i] = max(CE_ET[i], ET[theJob][M_OPTnum - 1])
    for i in range(0, CE_Tnum, 1):  # 到这都是不一样的
        if time_max < CE_ET[i]:
            time_max = CE_ET[i]
    # 到这时间已经计算了来自计算本身、通讯方面的消耗
    for i in range(0, Cnum, 1):
        if len(CloudLoad[i]) == 0:
            continue
        u_ratio = int(len(CloudLoad[i]) / 20.0 * 10)
        if (u_ratio > 10):
            u_ratio = 10
        time_expand = 0
        for star_iter in CloudLoad[i]:
            if (CE_ET[star_iter] - CE_ST[star_iter] > time_expand):
                time_expand = CE_ET[star_iter] - CE_ST[star_iter]
        energy += EnergyList[u_ratio] * time_expand / 1000.0   # 云服务器开着就费电， energy list存的就是多少比率占用多少的意思
    for i in range(0, Enum, 1):
        if (len(EdgeLoad[i]) == 0):
            continue
        u_ratio = int(len(EdgeLoad[i]) / 6.0 * 10)
        if (u_ratio > 10):
            u_ratio = 10
        time_expand = 0
        for star_iter in EdgeLoad[i]:
            if CE_ET[star_iter] - CE_ST[star_iter] > time_expand:
                time_expand = CE_ET[star_iter] - CE_ST[star_iter]
        energy += EnergyList[u_ratio] * time_expand / 1000.0  # 边缘服务器开着就费电

    obj_end = timeit.default_timer()

    fitness = time_max + energy
    if is_collect:
        if not isACO:  # ACO会多2维
            with open(buffer_path, 'a') as file:
                for eve in var:
                    print(round(eve, 5),  file=file, end=',')
                print(fitness, file=file)
    print(obj_end - obj_start, ',')

    if is_return_info:
        return fitness, time_max, energy
    else:
        return fitness
