import numpy as np
import networkx as nx
from line import embedding_main
import time
import os

seed = 123
np.random.seed(seed)
class Env:

    def __init__(self, mainPath):
        self.mainPath = mainPath
        self.dim = 64 + 2
        self.graph_dim = 64
        self.nodeNum = 4941
        self.maxSeedsNum = 50
        # self.networkName = 'power'
        self.networkName = 'GN'
        # self.nameList = ['../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.00.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.05.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.10.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.15.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.20.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.25.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.30.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.35.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.40.txt', '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.45.txt',
        #                  '../data/'+self.networkName+'/' + str(self.nodeNum) + '/0.50.txt']
        self.nameList = ['../data/'+self.networkName+'/network1.txt', '../data/'+self.networkName+'/network2.txt',
                          '../data/'+self.networkName+'/network3.txt', '../data/'+self.networkName+'/network4.txt',
                          '../data/'+self.networkName+'/network5.txt', '../data/'+self.networkName+'/network6.txt',
                          '../data/'+self.networkName+'/network7.txt', '../data/'+self.networkName+'/network8.txt',
                          '../data/'+self.networkName+'/network9.txt', '../data/'+self.networkName+'/network10.txt']
        # self.nameList = ['../data/'+self.networkName+".txt"]

        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.graphIndex = -1
        self.nextGraph()

    # 由有向无权图构造有向有权图
    def constrctGraph(self,edges):
        graph = nx.DiGraph()
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
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))
        self.nodeNum = len(self.graph.nodes())
        self.seeds = set()
        self.influence = 0
        self.embedInfo = self.getembedInfo()
        self.localInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded
        self.oneHopInfluenceList = np.zeros(self.nodeNum)-1  # init with -1, which means this node has not been recorded

    def reset(self):
        self.seeds = set([])
        self.influence = 0
        return self.seeds2input(self.seeds)

    def step(self, node):
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
        return state, reward, isDone

    # seeds -> (n+1,d)
    def seeds2input(self,seeds):
        input = np.array(self.embedInfo)
        flagList = np.array([])
        degreeList = np.array([])
        for i in range(self.nodeNum):
            degreeList = np.append(degreeList, self.graph.out_degree[i])
            if i in seeds:
                flagList = np.append(flagList, 0)
            else:
                flagList = np.append(flagList, 1)

        flagList = flagList.reshape((self.nodeNum,1))
        degreeList = degreeList.reshape((self.nodeNum, 1))
        input = np.hstack((degreeList, input))
        input = np.hstack((flagList,input))

        return input

    # seeds -> (n+1)*d
    def getembedInfo(self):
        try:
            embedInfo = np.loadtxt("../data/e/" + self.networkName + str(self.graphIndex))
            # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + str(self.graphIndex))
            np.savetxt("result/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
            # embedInfo = np.loadtxt("../data/embedding/" + self.networkName + "matrix" + str(self.graphIndex))
        except:
            edges = self.edges.copy()
            path = '../data/graphwithVN.txt'
            np.savetxt(path, edges)
            os.system('python line.py --num_nodes ' + str(self.nodeNum) + ' --embedding_dim ' + str(self.graph_dim))
            embedInfo = np.loadtxt("../data/_embedding.txt")
            np.savetxt("../data/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
            np.savetxt(self.mainPath + "/embedding/" + self.networkName + str(self.graphIndex), embedInfo)
        return embedInfo

    def getInfluence(self, S):
        influence = 0
        for s in S:
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
        if self.localInfluenceList[node] >= 0:
            return self.localInfluenceList[node]

        result = 1
        Cu = set(self.graph.successors(node))
        for c in Cu:
            temp = self.getOneHopInfluence(c)
            Cc = set(self.graph.successors(c))
            if node in Cc:      # if egde (c,node) exits
                 temp = temp - self.graph[c][node]['weight']
            temp = temp * self.graph[node][c]['weight']
            result += temp
        self.localInfluenceList[node] = result
        return result

    # input a node
    def getOneHopInfluence(self, node):
        if self.oneHopInfluenceList[node] >= 0:
            return self.oneHopInfluenceList[node]

        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']

        self.oneHopInfluenceList[node] = result
        return result

    # input a set of nodes
    def getEpsilon(self, S):
        result = 0

        for s in S:
            Cs = set(self.graph.successors(s))  # neighbors of node s
            S1 = Cs - S
            for c in S1:
                Cc = set(self.graph.successors(c))  # neighbors of node c
                S2 = Cc & S
                # S2 = S2 - {s}
                result += (0.01 * len(S2))
                # for d in S2:
                #     result += self.graph[s][c]['weight'] * self.graph[c][d]['weight']
        return result
