import networkx as nx
import numpy as np
import pickle

class DBLPDataLoader:
    def __init__(self, graph_file, num_nodes=128):
        # self.g = create_graph_edgelist(graph_file,num_nodes)# 猜测是从graph中选出指定的num_nodes个节点的
        # self.g = nx.read_gpickle(graph_file)---networkx 1.9之前的用法
        # self.g = gpickle.read_gpickle(graph_file)---networkx 2.6之前的用法
        # with open(r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\ERL-d1\code\data\new_graph.pkl', 'rb') as f:
        #     self.g = pickle.load(f)# 从文件中读取图
        with open(graph_file, 'rb') as f:
            self.g = pickle.load(f)# 从文件中读取图
            
        # # 要从g中选出num_nodes个节点
        # nodes = list(self.g.nodes)[:num_nodes]
        # # 创建一个新的子图
        # self.g = self.g.subgraph(nodes)
        
        # Graph对象，边、节点
        self.num_of_nodes = self.g.number_of_nodes()
        self.num_of_edges = self.g.number_of_edges()
        self.edges_raw = self.g.edges(data=True)
        self.nodes_raw = self.g.nodes(data=True)

        self.edge_distribution = np.array([attr['weight'] for _, _, attr in self.edges_raw], dtype=np.float32)# 提取出所有边的权重
        self.edge_distribution /= np.sum(self.edge_distribution)
        self.edge_sampling = AliasSampling(prob=self.edge_distribution)
        self.node_negative_distribution = np.power(
            np.array([self.g.degree(node, weight='weight') for node, _ in self.nodes_raw], dtype=np.float32), 0.75)
        self.node_negative_distribution /= np.sum(self.node_negative_distribution)
        self.node_sampling = AliasSampling(prob=self.node_negative_distribution)

        self.node_index = {}
        self.node_index_reversed = {}
        for index, (node, _) in enumerate(self.nodes_raw):
            self.node_index[node] = index
            self.node_index_reversed[index] = node
        self.edges = [(self.node_index[u], self.node_index[v]) for u, v, _ in self.edges_raw]

    def fetch_batch(self, batch_size=16, K=10, edge_sampling='atlas', node_sampling='atlas'):
        if edge_sampling == 'numpy':
            edge_batch_index = np.random.choice(self.num_of_edges, size=batch_size, p=self.edge_distribution)
        elif edge_sampling == 'atlas':
            edge_batch_index = self.edge_sampling.sampling(batch_size)
        elif edge_sampling == 'uniform':
            edge_batch_index = np.random.randint(0, self.num_of_edges, size=batch_size)
        u_i = []
        u_j = []
        label = []
        for edge_index in edge_batch_index:
            edge = self.edges[edge_index]
            if self.g.__class__ == nx.Graph:
                if np.random.rand() > 0.5:      # important: second-order proximity is for directed edge
                    edge = (edge[1], edge[0])
            u_i.append(edge[0])
            u_j.append(edge[1])
            label.append(1)
            for i in range(K):
                while True:
                    if node_sampling == 'numpy':
                        negative_node = np.random.choice(self.num_of_nodes, p=self.node_negative_distribution)
                    elif node_sampling == 'atlas':
                        negative_node = self.node_sampling.sampling()
                    elif node_sampling == 'uniform':
                        negative_node = np.random.randint(0, self.num_of_nodes)
                    if not self.g.has_edge(self.node_index_reversed[negative_node], self.node_index_reversed[edge[0]]):
                        break
                u_i.append(edge[0])
                u_j.append(negative_node)
                label.append(-1)
        return u_i, u_j, label

    def embedding_mapping(self, embedding):
        return {node: embedding[self.node_index[node]] for node, _ in self.nodes_raw}

class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n# 除以总和再*总数量
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []# 分别存储超过和不超过1的元素的索引
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res