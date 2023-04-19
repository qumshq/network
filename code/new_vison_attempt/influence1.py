import numpy as np
import networkx as nx
# from line import embedding_main
# import time
import os

seed = 123
np.random.seed(seed)
class Env:

    def __init__(self, mainPath):
        self.mainPath = mainPath
        self.nodeNum = 4941#4941是怎么得到的？--在读取txt文件之后会统计txt文件的节点个数更新此值
        self.maxSeedsNum = 10# 选取的种子节点数量
        
        # 单个图片的namelist
        # 物流数据
        self.nameList = ['../data/single_graphdata/Fried_edgelist_224.txt']
        # SDNE编码
        self.embedList = ['../data/single_graphdata/Fried_edgelist_224_sdne_100dim.txt']
        self.dim = 100 + 2# --how to get this?
        self.graph_dim = 100 # 64是line编码得到的特征，如果换用sden模型的话每个节点的特征数就是：2
        # # line编码
        # self.embedList = ['../data/single_graphdata/Fried_edgelist_224line_1.txt']
        # self.dim = 64 + 2
        # self.graph_dim = 64
        
# =============================================================================
#         # # 选择编码好的图像并按照是使用line或者sdne采用不同的特征dim
#         # self.use_dif_enbed = 'embedding_result_sdne_SF'# 使用sdne编码
#         # # self.use_dif_enbed = 'embedding_result_line_SF'# 使用line编码
#         # # self.use_dif_enbed = 'Karate_graph_transform/embedding_line'# 使用line编码
#         # # self.use_dif_enbed = 'Karate_graph_transform/embedding_result'# 使用sdne编码
#         # if self.use_dif_enbed == 'embedding_result_sdne_SF' or self.use_dif_enbed == 'Karate_graph_transform/embedding_result':
#         #     print('embeded by model SDNE')
#         #     self.dim = 2 + 2
#         #     self.graph_dim = 2
#         # else:
#         #     print("embeded by model LINE")
#         #     self.dim = 64 + 2# --how to get this?
#         #     self.graph_dim = 64 # 64是line编码得到的特征，如果换用sden模型的话每个节点的特征数就是：2
#             
#         # # 选择图像对象，即编码之前的图像，与use_dif_enbed相对应
#         # # self.networkName = 'power'
#         # # self.networkName = 'GN'
#         # self.networkName = 'SF_NetData'
#         # # self.networkName = 'Karate_graph_transform'
#         # """
#         # GN和LFR中的txt文件代表的是128个节点，多条边的无向图，对于每条边都存储了a-b和b-a两次
#         # """
#         # """疑问：此处的文件list迭代使用每一张图作为训练，只有最后一张图作为训练并保存结果，是否不需要前面的图像迭代训练"""
#         # # self.nameList = ['../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.00.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.05.txt',
#         # #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.10.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.15.txt',
#         # #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.20.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.25.txt',
#         # #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.30.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.35.txt',
#         # #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.40.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.45.txt',
#         # #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.50.txt']
#         # if self.networkName == 'SF_NetData':
#         #     self.nameList = ['../data/'+self.networkName+'/1_edgelist', '../data/'+self.networkName+'/2_edgelist',
#         #                       '../data/'+self.networkName+'/3_edgelist', '../data/'+self.networkName+'/4_edgelist',
#         #                       '../data/'+self.networkName+'/5_edgelist', '../data/'+self.networkName+'/6_edgelist',
#         #                       '../data/'+self.networkName+'/7_edgelist', '../data/'+self.networkName+'/8_edgelist',
#         #                       '../data/'+self.networkName+'/9_edgelist']
#         #     self.embedList = ['../data/'+self.use_dif_enbed+'/_embedding0.txt', '../data/'+self.use_dif_enbed+'/_embedding1.txt',
#         #                   '../data/'+self.use_dif_enbed+'/_embedding2.txt', '../data/'+self.use_dif_enbed+'/_embedding3.txt',
#         #                   '../data/'+self.use_dif_enbed+'/_embedding4.txt', '../data/'+self.use_dif_enbed+'/_embedding5.txt',
#         #                   '../data/'+self.use_dif_enbed+'/_embedding6.txt', '../data/'+self.use_dif_enbed+'/_embedding7.txt',
#         #                   '../data/'+self.use_dif_enbed+'/_embedding8.txt']
#         # elif self.networkName == 'GN':
#         #     self.nameList = ['../data/'+self.networkName+'/network1.txt', '../data/'+self.networkName+'/network2.txt',
#         #                       '../data/'+self.networkName+'/network3.txt', '../data/'+self.networkName+'/network4.txt',
#         #                       '../data/'+self.networkName+'/network5.txt', '../data/'+self.networkName+'/network6.txt',
#         #                       '../data/'+self.networkName+'/network7.txt', '../data/'+self.networkName+'/network8.txt',
#         #                       '../data/'+self.networkName+'/network9.txt', '../data/'+self.networkName+'/network10.txt']
#         #     self.embedList = ['../data/'+self.use_dif_enbed+'/_embedding0.txt', '../data/'+self.use_dif_enbed+'/_embedding1.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding2.txt', '../data/'+self.use_dif_enbed+'/_embedding3.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding4.txt', '../data/'+self.use_dif_enbed+'/_embedding5.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding6.txt', '../data/'+self.use_dif_enbed+'/_embedding7.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding8.txt', '../data/'+self.use_dif_enbed+'/_embedding9.txt']
#         # else:
#         #     self.nameList = ['../data/'+self.networkName+'/'+self.networkName+'/1change.edgelist', '../data/'+self.networkName+'/'+self.networkName+'/2change.edgelist',
#         #                       '../data/'+self.networkName+'/'+self.networkName+'/3change.edgelist', '../data/'+self.networkName+'/'+self.networkName+'/4change.edgelist',
#         #                       '../data/'+self.networkName+'/'+self.networkName+'/5change.edgelist', '../data/'+self.networkName+'/'+self.networkName+'/6change.edgelist',
#         #                       '../data/'+self.networkName+'/'+self.networkName+'/7change.edgelist', '../data/'+self.networkName+'/'+self.networkName+'/8change.edgelist',
#         #                       '../data/'+self.networkName+'/'+self.networkName+'/9change.edgelist', '../data/'+self.networkName+'/'+self.networkName+'/10change.edgelist']
#         #     self.embedList = ['../data/'+self.use_dif_enbed+'/_embedding0.txt', '../data/'+self.use_dif_enbed+'/_embedding1.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding2.txt', '../data/'+self.use_dif_enbed+'/_embedding3.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding4.txt', '../data/'+self.use_dif_enbed+'/_embedding5.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding6.txt', '../data/'+self.use_dif_enbed+'/_embedding7.txt',
#         #                       '../data/'+self.use_dif_enbed+'/_embedding8.txt', '../data/'+self.use_dif_enbed+'/_embedding9.txt']
#         # # self.nameList = ['../data/'+self.networkName+".txt"]
# =============================================================================

        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        # 这里的list用于单个种子节点产生影响力的过程
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.graphIndex = -1# 用来记录目前遍历到namelist中的哪一个图了
        self.nextGraph()

    # 由有向无权图构造有向有权图
    # TODO：算法针对的是无向图，此处有无向图创建有向图的转换方式存疑
    def constrctGraph(self,edges):
        ## 输入：无权图的边集edges，是一个弧头-弧尾的二维数组
        ## 输出：将所有边的权重均设置为自定义值p，这个p就是种子节点激活其他节点的概率，返回一个nx图像G和边集数组edge1
        """以图像GN为例，发现txt文件中边是成对存在的，即按理说表示的是无向图，但是1-2边存在则对应的有2-1边"""
        graph = nx.DiGraph()# 创建一个nx图对象
        graphP = nx.DiGraph()

        for u, v in edges:
            u = int(u)
            v = int(v)
            graph.add_edge(u, v)

        nodesList = list(graph.nodes())     # 把点的id映射到0~n-1
        nodeMap = dict()
        index = 0
        for node in nodesList:
            nodeMap[node] = index
            index += 1

        edges1 = np.array([])
        indegree = graph.in_degree()
        outdegree = graph.out_degree()
        for edge in graph.edges():
            u, v = edge

            # 采用不同的方式给边赋予权重
            # p = outdegree[u] / (outdegree[u] + indegree[v])
            # p = (outdegree[u] + indegree[u]) / (outdegree[u] + indegree[u] + outdegree[v] + indegree[v])
            # p = 1 / (outdegree[v] + indegree[v])
            # p = 1 / (indegree[v])
            p = 0.1

            u = nodeMap[u]
            v = nodeMap[v]

            if not graphP.has_edge(u, v):
                edges1 = np.append(edges1, (u, v, p))
            graphP.add_edge(u, v, weight=p)


        edges1 = edges1.reshape((len(graphP.edges()), 3))
        return graphP, edges1

    def nextGraph(self):
        # 调用这个函数开始对下一张图进行训练，更新env对象中的节点和边，重新设置种子节点为空
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))# 读取txt文件的点和边，每个文件有128个点
        self.nodeNum = len(self.graph.nodes())
        self.seeds = set()# 空字典
        self.influence = 0
        self.embedInfo = self.getembedInfo()
        # 这里创建进程调用了line，使用line对文件，获取了graphwithVN.txt文件中的节点特征，但是在改变选种子图像的时候没有更新特征
        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded

    def reset(self):
        self.seeds = set([])
        self.influence = 0
        return self.seeds2input(self.seeds)

    def step(self, node):
        # 算法中每次选择Q值最高的非种子节点进入种子集
        # node是进化步骤中当前Q值最高的点，判断其是否已经在种子集中，否则加入种子集返回加入它的奖励值和状态
        state = None
        if node in self.seeds:
            print("choose repeated node!!!!!!!!!!!!!")
            state = self.seeds2input(self.seeds)
            return state, 0, False

        self.seeds.add(node)
        reward = self.getInfluence(self.seeds) - self.influence

        self.influence += reward

        isDone = False
        if len(self.seeds) == self.maxSeedsNum:
            isDone = True

        state = self.seeds2input(self.seeds)
        return state, reward, isDone# state:第一列表示是否为种子节点，第二列为度，后64列为特征

    # seeds -> (n+1,d)
    def seeds2input(self,seeds):# 把种子节点转化为DQN的输入形式：[度,S,H]
    # TODO：考虑如果转为SDNE编码之后是否还需要增加节点度这一列信息
        input = np.array(self.embedInfo)
        flagList = np.array([])# 节点是否在seeds集中，0表示在
        degreeList = np.array([])
        for i in range(self.nodeNum):
            degreeList = np.append(degreeList, self.graph.out_degree()[i])#出度
            if i in seeds:
                flagList = np.append(flagList, 0)
            else:
                flagList = np.append(flagList, 1)

        flagList = flagList.reshape((self.nodeNum,1))
        degreeList = degreeList.reshape((self.nodeNum, 1))
        input = np.hstack((degreeList, input))
        # input = np.hstack((flagList,input))
        # input = np.hstack((degreeList, np.transpose(input)))
        input = np.hstack((flagList,input))

        return input

    # seeds -> (n+1)*d
    def getembedInfo(self):
        embedInfo = np.loadtxt(self.embedList[self.graphIndex])# 将在此处调用line模型改写成了读取已经编码好的模型---时间上不具有实验结果的说服力
        # try:
        #     embedInfo = np.loadtxt("../data/e/" + self.networkName + str(self.graphIndex))
        #     # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + str(self.graphIndex))
        #     np.savetxt("result/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
        #     # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + "matrix" + str(self.graphIndex))
        # except:
        #     """问题：编码获取节点特征的是graphwithVN.txt图文件，与我们选择种子节点的文件不一致，是否是因为理解错误
        #     实际上应该是每次nextgraph的时候调用这个函数对当前图像进行line编码，然后返回编码特征
        #     ---解决：每次调用nextgraph的时候更新graphwithVN文件中的edges，然后对这个图像进行编码
        #     结果存储到_embedding_embedding，只是每次os.system似乎是创建新进程，因此不等图像更新就会使用原来的特征
        #     continue了。
        #     """
        #     edges = self.edges.copy()
        #     path = '../data/graphwithVN.txt'
        #     np.savetxt(path, edges)
        #     os.system('python line.py --num_nodes ' + str(self.nodeNum) + ' --embedding_dim ' + str(self.graph_dim))
        #     # 会创建一个子进程执行这个程序
        #     #system函数可以将字符串转化成命令在服务器上运行，这里调用line文件，获得dim*num_nodes的特征编码矩阵
        #     embedInfo = np.loadtxt("../data/_embedding.txt")# 这个文件的作业是graphwithVN.txt使用line编好码后得到的特征文件
        #     np.savetxt("../data/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
        #     np.savetxt(self.mainPath + "/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
        return embedInfo

    def getInfluence(self, S):
        # TODO：考虑influence的公式
        # 输入：种子集S
        # 输出：影响力，计算公式由三部分组成 所有节点的影响扩散-冗余的1跳影响力-冗余的2跳影响力
        """
        论文公式理解：
        1.对每个节点单独使用IC模型进行传播得到单个节点的影响力，求和
        2.减去冗余的1跳影响力和2跳影响力
        """
        influence = 0
        for s in S:# 第1部分，所有节点的影响力扩散
            influence += self.getLocalInfluence(s)

        influence -= self.getEpsilon(S)

        for s in S:
            Cs = set(self.graph.successors(s))
            S1 = S & Cs
            for s1 in S1:
                influence -= self.graph[s][s1]['weight'] * self.getOneHopInfluence(s1)
        return influence

    # one node local influence
    def getLocalInfluence(self, node):
        if self.localInfluenceList[node] >= 0:# 列表初始化为-1，>=0表示这个节点已经被其他节点激活
            return self.localInfluenceList[node]

        result = 1
        Cu = set(self.graph.successors(node))
        # Cu：当前种子节点node的相邻节点
        # graph.successors(node)返回所有已node为起点的后继节点列表
        for c in Cu:
            temp = self.getOneHopInfluence(c)# 这个结果应该是论文公式中的tao
            Cc = set(self.graph.successors(c))# c的邻接点，即与种子node有2跳相连的点
            if node in Cc:      # if egde (c,node) exits
                 temp = temp - self.graph[c][node]['weight']# 减去2跳后回到种子的边
            temp = temp * self.graph[node][c]['weight']
            result += temp
        self.localInfluenceList[node] = result
        return result

    # input a node
    def getOneHopInfluence(self, node):# 返回论文中的tao，即对1+输入节点所有邻边权和
        if self.oneHopInfluenceList[node] >= 0:
            return self.oneHopInfluenceList[node]

        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']

        self.oneHopInfluenceList[node] = result
        return result# 所有以之为起点的边的权重之和+1，这个结果应该是论文公式中的tao

    # input a set of nodes
    def getEpsilon(self, S):
        result = 0

        for s in S:
            Cs = set(self.graph.successors(s))  # neighbors of node s
            S1 = Cs - S# 当前种子的邻居中不属于种子集的
            for c in S1:
                Cc = set(self.graph.successors(c))  # neighbors of node c
                S2 = Cc & S
                # S2 = S2 - {s}
                result += (0.01 * len(S2))
                # for d in S2:
                #     result += self.graph[s][c]['weight'] * self.graph[c][d]['weight']
        return result