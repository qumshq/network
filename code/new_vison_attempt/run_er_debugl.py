import numpy as np, os, time, sys, random

import torch
from torch.optim import Adam
import torch.nn as nn
import replay_memory

import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from influence1 import Env
from SSNE import SSNE
from DQN_Model import DQN
import mod_utils as utils

import os

class Agent:
    def __init__(self, env):
        self.is_cuda = False#True，是否使用cuda加速
        self.is_memory_cuda = False#True
        self.batch_size = 512
        self.use_done_mask = True
        self.pop_size = 100# 种群数量？
        self.buffer_size = 10000
        self.randomWalkTimes = 20
        self.learningTimes = 3
        self.action_dim = None  # Simply instantiate them here, will be initialized later
        self.dim = env.dim

        self.env = env
        self.evolver = SSNE(self.pop_size)#稀疏网络中链路预测的有效节点表示？
        self.evalStep = env.maxSeedsNum // 10    # step num during evaluation

        #Init population
        self.pop = []# 种群，种群中每一个个体为一个DQN网络参数，
        for _ in range(self.pop_size):
            self.pop.append(DQN(self.dim))
        self.all_fitness = []

        #Turn off gradients and put in eval mode
        for dqn in self.pop:
            dqn.eval()# 用于测试，锁定参数，否则当运行网络时即便不train参数也会改变

        #Init RL Agent
        self.rl_agent = DQN(self.dim)# rl_agent也是一个DQN网络对象
        self.gamma = 0.8            # discount rate
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=0.9995, last_epoch=-1)
        self.loss = nn.MSELoss()
        self.replay_buffer = replay_memory.ReplayMemory(self.buffer_size)
        # 用于存储一组组的state(包括选了哪些种子), action, next_state, reward, done的类对象，是强化学习的经验集

        #Trackers
        self.num_games = 0; self.num_frames = 0; self.gen_frames = 0

    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.is_cuda: reward = reward.cuda()
        if self.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)# 将done写成张量从形式
            if self.is_cuda: done = done.cuda()

        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, store_transition=True):# 每次传入一个DQN网络进行训练
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        # if self.is_cuda:
        #     state = state.cuda()
        done = False
        seeds = []
        while not done:# 没有找够50个种子之前
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            Qvalues = net.forward(state)
            Qvalues = Qvalues.reshape((Qvalues.numel(),))
            sorted, indices = torch.sort(Qvalues, descending=True)# 按照Q值排序并保存对应索引，Q是对应每一个节点给出的一个分数

            actionNum = 0

            for i in range(state.shape[1]):# state.shape[1]不就是点个数吗
                if state[0][indices[i]][0].item() == 1:         # choose node that is not seed--也即state中集成了1个表示是否为节点，一个表示度，64个特征值
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])

                    next_state, reward, done = self.env.step(actionInt)  # Simulate one step in environment

                    next_state = utils.to_tensor(next_state).unsqueeze(0)
                    if self.is_cuda:
                        next_state = next_state#.cuda()
                    total_reward += reward
                    if store_transition: self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
                    state = next_state

                    if actionNum == self.evalStep or done:          # finish after self.evalStep steps
                        break
            # end of for
        # end of while
        if store_transition: self.num_games += 1

        return total_reward, seeds

    def randomWalk(self):# 随机选出一组种子节点，返回总的奖励，并将途径的状态add_experience
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        # if self.is_cuda:
        #     state = state.cuda()
        done = False
        actionList = [i for i in range(self.env.nodeNum)]# 0到当前图像的节点个数
        actionIndex = 0
        random.shuffle(actionList)# 乱序
        while not done:
            self.num_frames += 1##??
            self.gen_frames += 1##??
            actionInt = actionList[actionIndex]
            action = torch.tensor([actionInt])
            next_state, reward, done = self.env.step(actionInt)  # Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            # if self.is_cuda:
            #     next_state = next_state.cuda()
            total_reward += reward
            self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
            state = next_state
            actionIndex += 1
        self.num_games += 1
        return total_reward

    def rl_to_evo(self, rl_net, evo_net):# gess:把rl_ent的参数赋值赋值给evo
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    def evaluate_all(self):
        self.all_fitness = []
        t1 = time.time()
        for net in self.pop:
            fitness, _ = self.evaluate(net)
            self.all_fitness.append(fitness)


        t2 = time.time()
        print("evaluate finished.    cost time:", t2 - t1)

    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################

        # NeuroEvolution's probabilistic selection and recombination step
        t1 = time.time()
        for _ in range(self.randomWalkTimes):
            self.randomWalk()# 随机选出一组种子节点，返回总的奖励，并保存途径状态，是强化学习对环境的探索
        best_train_fitness = max(self.all_fitness)

        new_pop = self.evolver.epoch(self.pop, self.all_fitness)# 新的种群
        print("new_pop_num:", len(new_pop))
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        t2 = time.time()
        print("epoch finished.    cost time:", t2 - t1)

        # rl learning step
        t1 = time.time()
        for _ in range(self.learningTimes):
            # worst_index = self.all_fitness.index(min(self.all_fitness))
            index = random.randint(len(self.pop)//2, len(self.pop)-1)
            self.rl_to_evo(self.pop[0], self.rl_agent)# pop[0]应当是目前的最佳种群，把最佳种群的参数复制给rl
            if len(self.replay_buffer) > self.batch_size * 2:
                transitions = self.replay_buffer.sample(self.batch_size)# 从经验集中采样batchsize个
                batch = replay_memory.Transition(*zip(*transitions))
                self.update_parameters(batch)
                self.rl_to_evo(self.rl_agent, self.pop[index])
                fitness, _ = self.evaluate(self.rl_agent, True)#evaluate返回的是排好序的
                self.all_fitness[index] = fitness
                print('Synch from RL --> Nevo')
        t2 = time.time()
        print("learning finished.    cost time:", t2 - t1)
        # self.scheduler.step()
        return best_train_fitness, sum(self.all_fitness)/len(self.all_fitness), self.rl_agent, self.pop[0:len(self.pop)//10]

    def update_parameters(self, batch):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = None
        if self.use_done_mask: done_batch = torch.cat(batch.done)
        state_batch.volatile = False; next_state_batch.volatile = True; action_batch.volatile = False

        # Load everything to GPU if not already
        # if self.is_cuda:
        #     self.rl_agent.cuda()
        #     state_batch = state_batch.cuda(); next_state_batch = next_state_batch.cuda(); action_batch = action_batch.cuda(); reward_batch = reward_batch.cuda()
        #     if self.use_done_mask: done_batch = done_batch.cuda()

        currentList = torch.Tensor([])
        currentList = torch.unsqueeze(currentList, 1)#.cuda()
        targetList = torch.Tensor([])
        targetList = torch.unsqueeze(targetList, 1)#.cuda()
        # DQN Update
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch, done_batch):
            target = torch.Tensor([reward])
            if not done:
                next_q_values = self.rl_agent.forward(next_state)
                pred, idx = next_q_values.max(0)
                target = reward + self.gamma * pred

            target_f = self.rl_agent.forward(state)

            current = target_f[action]
            current = torch.unsqueeze(current, 1)
            target = torch.unsqueeze(target, 1)#.cuda()
            currentList = torch.cat((currentList, current), 0)
            targetList = torch.cat((targetList, target), 0)

        self.optim.zero_grad()
        dt = self.loss(currentList, targetList)
        dt.backward()
        nn.utils.clip_grad_norm(self.rl_agent.parameters(), 10000)
        self.optim.step()

        # Nets back to CPU if using memory_cuda
        # if self.is_memory_cuda and not self.is_cuda:
        #     self.rl_agent.cpu()

    def get_offspring(self, pop, fitness_evals, new_pop, new_fitness_evals):
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        for i in range(len(pop)):
            all_pop.append(pop[i])
            fitness.append(fitness_evals[i])
        for i in range(len(new_pop)):
            all_pop.append(new_pop[i])
            fitness.append(new_fitness_evals[i])

        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__)
        index_rank.reverse()
        for i in range(len(pop) // 2):
            offspring.append(all_pop[index_rank[i]])
            offspring_fitness.append(fitness[index_rank[i]])

        randomNum = len(all_pop) - len(pop) // 2
        randomList = list(range(randomNum))
        random.shuffle(randomList)
        for i in range(len(pop) // 2, len(pop)):
            index = randomList[i-len(pop) // 2]
            offspring.append(all_pop[index])
            offspring_fitness.append(fitness[index])
            ...

        return offspring, offspring_fitness

    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        print(out)

def getResultPath():
    for i in range(1,21):
        path = "result/" + str(i) + "/"
        if os.path.exists(path):
            continue
        os.makedirs(path)# 创建文件夹
        os.makedirs(path + "/embedding/")
        os.makedirs(path + "/model/")
        return path# 每次调用都只在result文件夹下创建一个子文件夹


if __name__ == "__main__":
    #Create Env
    t1 = time.time()
    mainPath = getResultPath()# 创建子文件夹用于存储结果的模型和图

    env = Env(mainPath)# mainPath用于传递存储result的路径
    # Random seed setting随机种子
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #Create Agent
    agent = Agent(env)

    next_save = 100; time_start = time.time()
    print("Start training")
    maxList = np.array([])
    resultList = np.array([])
    timeList = np.array([])
    for graphIndex in range(1):#len(agent.env.nameList)):# 遍历每一个含图的文件，对每一个图文件依次得到影响、时间、种子集
        for i in range(1):#00):#:        # Generation
            if i == 0:
                agent.evaluate_all()# 对于agent中的每个对象按照DQN网络的Q值选出一个种子节点集，将返回的total_reward加入到list成员all_fitness
            print("=================================================================================")
            print(graphIndex, "th graph      Generation:", i, "    ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            best_train_fitness, average, rl_agent, elitePop = agent.train()#rl_agent是我们训练的目标DQN
            print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness, ' Avg:', average)
            maxList = np.append(maxList, best_train_fitness)
            #Save Policy

            torch.save(rl_agent, mainPath + "//model//rlmodel" + str(graphIndex))# 存储训练的DQN参数
            for eliteIndex in range(len(elitePop)):# elitePop是pop种群中前10%的个体
                torch.save(elitePop[eliteIndex], mainPath + "//model//elite_model" + str(graphIndex) + "_" + str(eliteIndex))
            np.savetxt(mainPath + "//maxList.txt", maxList)
            print("Progress Saved")

        fitness, seeds = agent.evaluate(agent.pop[0])
        print("best fitness:", fitness)
        print("seeds:", seeds)
        resultList = np.append(resultList, fitness)
        t2 = time.time()
        timeList = np.append(timeList, t2-t1)
        t1 = t2
        if graphIndex < len(agent.env.nameList) - 1:
            agent.env.nextGraph()# 如果当前不是训练list中的最后一张图，则调用next设置为下一张txt存储的图
            agent.replay_buffer = replay_memory.ReplayMemory(agent.buffer_size)
        else:
            break

    print("time cost:")
    agent.showScore(timeList)

    print("influence:")
    agent.showScore(resultList)
    np.savetxt(mainPath + "//timeList.txt", timeList)
    np.savetxt(mainPath + "//resultList.txt", resultList)
    np.savetxt(mainPath + "//seeds.txt", seeds)
    file = 'result//timeList.txt'# 所有图的总结
    with open(file, 'a') as f:
        for i in range(len(timeList)):
            f.write(str(timeList[i]) + "\t")
        f.write("\n")
    file = 'result//resultList.txt'

    with open(file, 'a') as f:
        for i in range(len(resultList)):
            f.write(str(resultList[i]) + "\t")
        f.write("\n")
