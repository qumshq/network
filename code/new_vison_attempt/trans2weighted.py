# 将txt文件转换文pkl文件
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()#使用tf v1，为了与模型兼容
# import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

"""注：需要构造成有权图"""
def constrctGraph(edges):
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

nameList = ['../data/来自line的数据/SF_N=1000']#,'../data/生成数据/ER_N=200_k=4_edgelist']
import networkx as nx
import pickle
for i,gra in enumerate(nameList):
    # graph, edges = constrctGraph(np.loadtxt(gra))
    graph = nx.read_edgelist(gra, nodetype=int)
    # 为每条边添加权重
    for u, v in graph.edges():
        graph.add_weighted_edges_from([(u, v, 1)])
    # 可视化--耗时较长且点过于的多密
    # nx.draw(graph, with_labels=True)
    # plt.show()
    # # 打开txt文件进行读取
    # with open(gra) as f:
    #     for line in f:
    #         # 去除行末尾的换行符，并将每行数据分割为两个节点名称
    #         edge = line.strip().split()
    #         # 将两个节点添加到图中
    #         graph.add_edge(edge[0], edge[1])
    
    # 处理图像中可能存在的孤立点---但由于实际读图进行选择种子的时候也不会读取孤立点，因此没必要补充孤立点
    # 重新说明：需要，如果不加入line编码中将没有孤立点的特征，因而报错
    gor = np.loadtxt(gra)
    for i in range(1000):
        if i in gor:
            continue
        else:
            print(f'有孤立点{i}')
            graph.add_node(i)
    
    # 将图像对象存储为pkl文件
    # with open(f'../data/Karate_graph_transform/pkl/network{i+1}.pkl', 'wb') as f:
    save_name = gra.split('/')[-1].split('.')[0]
    with open('../data/来自line的数据/'+save_name+'.pkl', 'wb') as f:
        pickle.dump(graph, f)

with open("../data/来自line的数据/SF_N=1000.pkl", "rb") as f:
    G = pickle.load(f)


















