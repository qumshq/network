from copy import deepcopy
import networkx as nx

pmn = 0.01  # 传播概率


def p(node1, node2):
    # 两个节点间的传播概率
    return pmn


def cal_sigmaS(S, g):
    """种子2-hop影响力评估指标"""
    G_t = deepcopy(g)# 赋值一个图对象
    res = 0
    influ = cal_influ(S, G_t)
    second_term_and_third_term = cal_secondandthird_term(S, G_t)
    res += influ - second_term_and_third_term
    return res


def cal_influ(S, g):  # 第一项
    influ = 0
    for node1 in S:
        influ += 1
        sum_temp = 0
        for node2 in list(g[node1]):
            sum_temp += 1
            sum_temp += len(list(g[node2])) * pmn
        influ += pmn * sum_temp
    return influ


def cal_secondandthird_term(S, g):  # 第二项和第三项
    second_term = 0
    third_term = 0
    for i in range(len(S)):
        seed = S[i]
        Cs = list(g[seed])
        Cs_simi = find_simi(S, Cs)
        sum_temp = 0
        for j in range(len(Cs_simi)):
            temp = 0
            node = Cs_simi[j]
            temp += 1
            temp += pmn * len(list(g[node]))
            temp -= pmn
            sum_temp += temp * pmn
        second_term += sum_temp
        # 第二项计算完毕
        Cs_dis_simi, Cs_simi, Cs_d = find_third(Cs, S, seed)
        for i in range(len(Cs_dis_simi)):
            for j in range(len(Cs_d)):
                if g.has_edge(Cs_dis_simi[i], Cs_d[j]):
                    third_term += pmn * pmn
        # 第三项计算完毕
    return second_term + third_term


def find_simi(S, Cs):
    Cs_simi = []
    for i in range(len(S)):
        temp_node = S[i]
        for j in range(len(Cs)):
            if (Cs[j] == temp_node):
                Cs_simi.append(temp_node)
    return Cs_simi


def find_third(Cs, S, seed):
    Cs_dis_simi = []
    Cs_d = []
    Cs_dis_simi_temp = deepcopy(Cs)
    for i in range(len(Cs_dis_simi_temp)):
        for j in range(len(S)):
            if (Cs_dis_simi_temp[i] == S[j]):
                Cs_dis_simi_temp[i] = -1
        if (Cs_dis_simi_temp[i] != -1):
            Cs_dis_simi.append(Cs_dis_simi_temp[i])
    Cs_simi = []
    for i in range(len(S)):
        temp_node = S[i]
        for j in range(len(Cs)):
            if Cs[j] == temp_node:
                Cs_simi.append(temp_node)
    for i in range(len(Cs_simi)):
        if Cs_simi[i] != seed:
            Cs_d.append(Cs_simi[i])
    return Cs_dis_simi, Cs_simi, Cs_d


# 示例
# S = [162, 12, 184, 89, 34, 8, 64, 140, 178, 26, 83, 86, 70, 191, 49, 88, 161, 27, 16, 123]  # 种子集合
S= [31, 125, 5, 77, 38, 73, 13, 96, 88, 135]# 14.87000000000000455
NetName = r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\ERL-d1\code\data\single_graphdata\Fried_edgelist_224.txt'  # 网络名字
G_ori = nx.read_adjlist(NetName, nodetype=int)
N = G_ori.number_of_nodes()
print(cal_sigmaS(S, G_ori))
