import numpy as np, os, time, random
import torch
from torch.optim import Adam
import torch.nn as nn
import replay_memory

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from SSNE import SSNE
from DQN_Model import DQN
import mod_utils as utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Agent:
    def __init__(self, env):
        self.is_cuda = False#True，是否使用cuda加速
        self.is_memory_cuda = False#True
        self.batch_size = 512
        self.use_done_mask = True
        self.pop_size = 50# 种群数量
        self.buffer_size = 10000
        self.randomWalkTimes = 20
        self.learningTimes = 10
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
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001)# 学习率太小是否会导致DQN更新慢而在很多轮都没有更新pop[0]
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

    def evaluate(self, net, store_transition=True):
        # @todo
        """每次传入一个DQN网络，在当前网络的参数下选出一组种子，
        这个种子集会更新到agent对象中的env环境里
        返回这个种子集及其影响力"""
        #这里是什么意思？？？？输出的值是什么意思？计算方法？
        ##
        ##
        ##
        total_reward = 0.0
        state = self.env.reset()
        # 清空当前env对象的seed集和influence集,返回初始化状态：没有种子节点的DQN输入[S,度,H]，第一列为S
        # DQN的输入对应的是一个环境的状态:[degree,seeds,heprosis]
        state = utils.to_tensor(state).unsqueeze(0)
        if self.is_cuda:
            state = state.cuda()
        done = False
        seeds = []
        while not done:              # 循环直到找出指定个数的种子
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            Qvalues = net.forward(state)# 节点个数*1大小的Q值列表，Q是对应每一个节点给出的一个分数
            Qvalues = Qvalues.reshape((Qvalues.numel(),))
            sorted, indices = torch.sort(Qvalues, descending=True)
            # 按照Q值排序并保存对应索引，sorted是经过排序的Q值列表，indices是当前位置的值在排序之前的索引

            actionNum = 0# 增加节点为种子这个操作的次数

            for i in range(state.shape[1]):# state.shape[1]不就是点个数吗--遍历每个点
                if state[0][indices[i]][0].item() == 1:         # choose node witch is not seed--也即state中集成了1个表示是否为节点，一个表示度，64个特征值
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])

                    next_state, reward, done = self.env.step(actionInt)  # Simulate one step in environment

                    next_state = utils.to_tensor(next_state).unsqueeze(0)
                    if self.is_cuda:
                        next_state = next_state.cuda()
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
        if self.is_cuda:
            state = state.cuda()
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
            # step的操作是将种子加入到env中的seeds集合，
            # 并返回加入该种子时获取的奖励reward，并且如果env中的种子集满了会返回done为true
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.is_cuda:
                next_state = next_state.cuda()
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
        # TODO
        """对种群self.pop中的每个DQN参数进行一次evaluate并"""
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
        # t1 = time.time()
        for _ in range(self.randomWalkTimes):
            self.randomWalk()# 随机选出一组种子节点，返回总的奖励，并保存途径状态，是强化学习对环境的探索
        best_train_fitness = max(self.all_fitness)

        new_pop = self.evolver.epoch(self.pop, self.all_fitness)# 新的种群，由多个DQN参数构成
        # print("new_pop_num:", len(new_pop))
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        # t2 = time.time()
        # print("epoch finished.    cost time:", t2 - t1)

        # rl learning step
        # t1 = time.time()
        for _ in range(self.learningTimes):# learningTimes每一轮对rl_agent参数进行训练并随机更换一个参数
            # worst_index = self.all_fitness.index(min(self.all_fitness))
            index = random.randint(len(self.pop)//2, len(self.pop)-1)# 种群一半到总种群数之间的一个随机数
            self.rl_to_evo(self.pop[0], self.rl_agent)# pop[0]是目前的最佳种群，把最佳种群的参数复制给rl
            if len(self.replay_buffer) > self.batch_size * 2:# self.replay_buffer是经验集
                transitions = self.replay_buffer.sample(self.batch_size)# 从经验集中采样batchsize个
                batch = replay_memory.Transition(*zip(*transitions))
                self.update_parameters(batch)
                self.rl_to_evo(self.rl_agent, self.pop[index])# 把经过改变的参数赋值给pop[index]
                fitness, _ = self.evaluate(self.rl_agent, True)#evaluate返回的是排好序的
                self.all_fitness[index] = fitness
                # print('Synch from RL --> Nevo')
        # t2 = time.time()
        # print("learning finished.    cost time:", t2 - t1)
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
        # with torch.no_grad():
        for state, action, reward, next_state, done in zip(state_batch, action_batch, reward_batch, next_state_batch, done_batch):
            target = torch.Tensor([reward])# 将reward作为标签
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
        nn.utils.clip_grad_norm_(self.rl_agent.parameters(), 10000)
        self.optim.step()

        # Nets back to CPU if using memory_cuda
        # if self.is_memory_cuda and not self.is_cuda:
        #     self.rl_agent.cpu()

    def get_offspring(self, pop, fitness_evals, new_pop, new_fitness_evals):
        """输入：原种群pop，原种群对应的影响力fitness_evals，新种群已经新种群对应的影响力"""
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        # 新旧种群均加入all_pop
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