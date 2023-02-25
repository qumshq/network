import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import networkx as nx
# 是否是生成节点的度

class Env:
    def __init__(self):
        self.maxSeedsNum = 10
        # graph_file = '../data/wiki.txt'
        # graph_file = '../data/wiki.txt'
        self.nameList = ['../data/GN/network1.txt', '../data/GN/network2.txt', '../data/GN/network3.txt',
                         '../data/GN/network4.txt', '../data/GN/network5.txt', '../data/GN/network6.txt',
                         '../data/GN/network7.txt', '../data/GN/network8.txt', '../data/GN/network9.txt',
                         '../data/GN/network10.txt']
        self.graphIndex = -1
        self.nextGraph()

    def constrctGraph(self, edges):
        graph = nx.DiGraph()
        graphP = nx.DiGraph()

        for u, v in edges:
            u = int(u)
            v = int(v)
            graph.add_edge(u,v)

        nodesList = list(graph.nodes())     # �ѵ��idӳ�䵽0~n-1
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
            graphP.add_edge(u, v, weight=p)
            edges1 = np.append(edges1, (u, v, p))

        edges1 = edges1.reshape((len(edges), 3))
        return graphP, edges1

    def nextGraph(self):
        self.graphIndex += 1
        graph_file = self.nameList[self.graphIndex]
        self.graph, self.edges = self.constrctGraph(np.loadtxt(graph_file))
        self.nodeNum = len(self.graph.nodes())
        self.seeds = set()
        self.influence = 0

    def reset(self):
        self.seeds = set([])
        self.influence = 0

    def degree(self):

        outdegree = self.graph.out_degree()
        # a = sorted(outdegree.items(), key=lambda i: i[1], reverse=True)
        a = sorted(dict(outdegree).items(), key=lambda i: i[1], reverse=True)
        self.seeds = set()
        for i in range(self.maxSeedsNum):
            self.seeds.add(a[i][0])
        print(self.graphIndex)
        print(self.getInfluence(self.seeds))


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
        result = 1
        Cu = set(self.graph.successors(node))
        for c in Cu:
            temp = self.getOneHopInfluence(c)
            Cc = set(self.graph.successors(c))
            if node in Cc:      # if egde (c,node) exits
                 temp = temp - self.graph[c][node]['weight']
            temp = temp * self.graph[node][c]['weight']
            result += temp

        return result

    # input a node
    def getOneHopInfluence(self, node):
        result = 1
        for c in self.graph.successors(node):
            result += self.graph[node][c]['weight']
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
                for d in S2:
                    result += self.graph[s][c]['weight'] * self.graph[c][d]['weight']
        return result


if __name__ == "__main__":
    env = Env()
    for i in range(10):
    # for i in range(9):
        env.degree()
        env.nextGraph()
