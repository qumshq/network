import tensorflow as tf
import numpy as np
import argparse
from line_model import LINEModel
from line_utils import GraphData
import pickle
import time
import networkx as nx


def embedding_main(graph_file):
    parser = argparse.ArgumentParser()
    #parser 命名空间，在cmd中运行python时可以给这些参数，不给就会按照default处理
    # print(parser)
    parser.add_argument('--embedding_dim',type=int, default=64)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=3500)
    parser.add_argument('--total_graph', default=True)
    parser.add_argument('--graph_file', default=graph_file)

    # 统计有多少个节点
    edges = np.loadtxt(graph_file)#form txt get array
    graph = nx.DiGraph()#创建nx有向图对象
    graph.add_weighted_edges_from(edges)#degs需要n*3为的数组表示有向图的起点-终点-权重
    nodeNum = len(graph.nodes())

    # +1 是对应着虚拟节点
    parser.add_argument('--num_nodes',type=int, default= 128)
    args = parser.parse_args()#args中存储了parser加入的变量
    print('args:',args)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        # test(args)
        print("00000")


def train(args):
    data_loader = GraphData(graph_file=args.graph_file,num_nodes=args.num_nodes)
    suffix = args.proximity
    args.num_of_nodes = data_loader.num_of_nodes
    model = LINEModel(args)
    with tf.Session() as sess:
        print(args)
        print('batches\tloss\tsampling time\ttraining_time\tdatetime')
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = args.learning_rate
        sampling_time, training_time = 0, 0
        for b in range(args.num_batches):
            t1 = time.time()
            u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
            t2 = time.time()
            sampling_time += t2 - t1
            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                if b% 1000 == 0:
                    print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            # if b % 1000 == 0 or b == (args.num_batches - 1):
            #     embedding = sess.run(model.embedding)
            #     normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            #     #print(normalized_embedding)
            #     pickle.dump(data_loader.embedding_mapping(normalized_embedding),
            #                 open('data/embedding_%s.pkl' % suffix, 'wb'))
            if b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                embedding_save_address = '../data/_embedding.txt'
                with open(embedding_save_address,'w') as f:
                    for i in range(args.num_nodes):
                        for j in range(args.embedding_dim):
                            f.write(str(normalized_embedding[i][j])+' ')
                        if i != (args.num_nodes-1):
                            f.write('\n')
                            print('called')


# def test(args):
#     pass

if __name__ == '__main__':
    # graph_file = '../data/test_graph.txt'
    graph_file = r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\line-master\data\co-authorship_graph.pkl'
    # 是否需要将所有的节点度算出并加入到最后一列
    # embedding_main(graph_file,32)
    embedding_main('../data/graphwithVN.txt')#这里应该有一个n*3维的矩阵，但是俺没有