import timeit
import random
from Multimethod import MultiMet
from Problems import CED_Schedule
import numpy as np
from surrogate.Surrogate_FCN import SurrogateFCN
from surrogate.Surrogate_CNN_flatten import SurrogateCNN
from surrogate.Surrogate_DNN import SurrogateDNN
from surrogate.Surrogate_RNN import SurrogateRNN
from surrogate.Surrogate_RCNN import SurrogateRCNN
from surrogate.Surrogate_Transformer import SurrogateTransformer
import torch
torch.set_num_threads(1)


import datetime
now = datetime.datetime.now()
import argparse
parser = argparse.ArgumentParser(description="Surrogate-assisted evolutionary algorithm")
parser.add_argument('-case', type=int, default=1, help='cases of 1(800) 2(900) 3(1000)')
parser.add_argument('-iss', type=int, default=1, help='1: open surrogate, 0: close')
parser.add_argument('-model', type=int, default=6, help='1: FCRN 2: DNN 3: CNN 4: RNN 5: RCNN 6: Transformer')
args = parser.parse_args()

SA_type_dict = {1: 'FCRN', 2: 'DNN', 3: 'CNN', 4: 'RNN', 5: 'RCNN', 6: 'Transformer'}
# hyparas for problem
CASE = args.case  # 1: 800+800, 2: 900+900, 3: 1000+1000
if CASE == 1:
    TNUM = 800
    ENUM = 700
    CNUM = 700
    DNUM = 400
    MOPT_NUM = 5
    load_name = 8
elif CASE == 2:
    TNUM = 900
    ENUM = 900
    CNUM = 900
    DNUM = 400
    MOPT_NUM = 5
    load_name = 9
else:
    TNUM = 1000
    ENUM = 1000
    CNUM = 1000
    DNUM = 600
    MOPT_NUM = 5
    load_name = 10

# EA hyparas for EA algorithms
MAXGEN = 100
POPSIZE = 10
is_surrogate = bool(args.iss)
is_watch_true_gbest = True  # 每一代是否查看真实的全局最优解
SA_type = args.model  # 1: FCRN 2: DNN 3: CNN

# Training surrogate
train_step = 20  # 多少代训练一次
is_online_train = True  # 是否在线训练代理模型
collect_gen = int(MAXGEN * 0.2)   # 前多少代一直收集 0.2
buffer_maxlen = 3000  # buffer队列最大长度
is_save_model = True  # 是否存储模型


if __name__ == "__main__":

    with open('./logs/running.txt', 'w') as f:
        pass
    with open('./logs/SA_GT_compare.txt', 'w') as f2:
        pass
    if is_surrogate:
        logpath = './logs/surrogate_' + SA_type_dict[SA_type] + '-' + str(now.now()).split('.')[0].replace(":", "-") + '.txt'
    else:
        logpath = './logs/surrogate_none-' + str(now.now()).split('.')[0].replace(":", "-") + '.txt'
    with open(logpath, 'w') as f3:
        pass
    root = './surrogate/model/'
    generation = 0
    record = list(range(int(MAXGEN)))
    # random.seed(1)
    # 求解器 100*2 + 100*5*2 = 1200
    solver = MultiMet(POPSIZE, TNUM * 2 + TNUM * MOPT_NUM * 2, 0, 1, CNUM, ENUM, DNUM, TNUM, TNUM, MOPT_NUM, CED_Schedule)
    solver.Initial(load_name)

    if SA_type == 1:
        SAmodel = SurrogateFCN(buffer_maxlen)
    elif SA_type == 2:
        SAmodel = SurrogateDNN(buffer_maxlen, TNUM*12)
    elif SA_type == 3:
        SAmodel = SurrogateCNN(buffer_maxlen, TNUM*12)
    elif SA_type == 4:
        SAmodel = SurrogateRNN(buffer_maxlen, TNUM*12)
    elif SA_type == 5:
        SAmodel = SurrogateRCNN(buffer_maxlen, TNUM*12)
    elif SA_type == 6:
        SAmodel = SurrogateTransformer(buffer_maxlen, TNUM*12)
    else:
        assert False
    # SAmodel = SurrogateFCN(buffer_maxlen)  # A class
    # SAmodel = SurrogateDNN(buffer_maxlen)
    # SAmodel = SurrogateCNN(buffer_maxlen)

    Gen_count = 0
    best = solver.gbest_fit

    t1 = timeit.default_timer()
    while generation+1 < MAXGEN:
        tmp = np.array(solver.pop)
        losses = []
        solver.DE(0.5, random.randint(0, 4), 0.5, 0, POPSIZE)
        # 评估并收集数据
        if is_surrogate:
            if generation == collect_gen:  # 开始第一次训练
                losses = SAmodel.train_model(True)
            generation_eval_data = solver.SAEvaluation(1, 0, POPSIZE, SAmodel.model, generation, collect_gen)
            # 将收集的GT数据添加进增量训练集中
            SAmodel.add_samples(generation_eval_data)
            # 增量式训练代理模型
            if generation % train_step == 0 and generation > collect_gen and is_online_train is True:
                losses = SAmodel.train_model(False)
                if is_save_model:
                    SAmodel.save_model(root, generation)
        else:
            solver.Evaluation(1, 0, POPSIZE)

        solver.pop_update(0, POPSIZE)
        if is_surrogate:  # gb一定要ground truth
            solver.SA_worst_and_best()
        else:
            solver.worst_and_best()
        solver.Elist()
        if solver.gbest_fit < best:
            Gen_count = 0
        else:
            Gen_count += 1
        generation += 1
        best = solver.gbest_fit
        t2 = timeit.default_timer()
        print('Current generation is', generation)
        with open(logpath, 'a') as f4:
            print('Gen:', generation, 'gb_fit:', solver.gbest_fit, 'time:', t2-t1,
                  'buffer len:', SAmodel.get_buffer_size(), 'losses:', losses, file=f4)
        record[generation] = solver.gbest_fit
    t3 = timeit.default_timer()
    print("----------------------------------------Generation = ",  generation)
    print("The best solution = ", solver.gbest_fit)
    print("Time = ", (t3 - t1), " s")
