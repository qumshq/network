# -*- coding: utf-8 -*-
import networkx as nx
from cul_inf import cal_sigmaS
import numpy as np

# path = '../data/生成数据/BA_N=200_k=4_edgelist'
# pathLsit = ['../data/生成数据/ER_N=200_k=4_edgelist','../data/生成数据/Fried_edgelist_224']
pathLsit = ['../data/生成数据2/WS_N=200_k=4_edgelist']
for path in pathLsit:
    # G = nx.read_adjlist(path, nodetype=int)
    G = nx.read_edgelist(path, nodetype=int)
    # 创建副本
    G_copy = G.copy()
    
    degree_dict = dict(G.degree(G.nodes()))# degree列表
    """
        攻击边:
        选取度最大的att_node_num个节点，每个节点按照其邻居的度大小尝试依次删除edge_for_node条边
        攻击节点:
        为了保持编码任然适用（即不需要度图像进行重新编码，攻击节点模拟为将节点的所有边remove）
    """
    # 获取度最大的几个边的节点编号
    att_node_num = 10# 选择攻击度最大的前10个节点
    edge_for_node = 3# 每个节点最多攻击3条边
    ed_moved = 0
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:att_node_num]
    for i,n in enumerate(sorted_nodes):
        neighbors = list(G.neighbors(n))
        ne_degree = dict(G.degree(neighbors))
        max_neb = sorted(ne_degree, key=ne_degree.get, reverse=True)[:edge_for_node+1]
        for k in max_neb:# 如果要攻击节点，则遍历删除neighbors
            if k == n:
                continue
            else:
                try:
                    G_copy.remove_edge(n, k)# 可能在之前的循环中已经将这条边删除
                    ed_moved = ed_moved+1
                except :
                    print(f"重复删除边{n}-{k}")
    # 将破坏后的图保存为边列表
    nx.write_edgelist(G_copy, path + "_removed.txt",data=False)
    
    G2 = nx.read_edgelist(path + "_removed.txt", nodetype=int)
    # 读取种子，进行影响力对比
    seed_path = r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\ERL-d1\code\new_vison_attempt\result\10 sdne 物流数据\seeds.txt'
    # S = np.loadtxt(seed_path)
    S = [189, 26, 2, 159, 8, 6, 3, 10, 13]
    inf_ori = cal_sigmaS(S, G)
    inf_de = cal_sigmaS(S, G_copy)
    inf_de2 = cal_sigmaS(S, G2)
    print(f'被破坏前影响力大小为：{inf_ori},移除了{ed_moved}条边之后影响力为：{inf_de},重新读取的影响力为:{inf_de2}')
