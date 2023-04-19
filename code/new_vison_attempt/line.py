# 读取pickle文件中存储的图，然后进行编码最终对每个节点得到embedding_dim(默认为128)个特征
# 最终得到一个n*embedding_dim的矩阵，将其存储在_embedding.txt文件中
# 这里使用的pickle图文件：使用node edge表示节点和边(无权图)，然后存储在pickle文件中
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()#使用tf v1，为了与模型兼容
# import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
# 自定义文件
from line_model import LINEModel
from line_utils import DBLPDataLoader

def main():
    # 命名空间，设置超参数变量
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=2)# 编码特征数
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--K', default=5)
    parser.add_argument('--proximity', default='second-order', help='first-order or second-order')
    parser.add_argument('--learning_rate', default=0.025)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--num_batches', default=50000)#300000，训练轮数
    parser.add_argument('--total_graph', default=True)
    """用于编码的图像路径"""
    
    # 读取转换为有向图的图片pkl文件
    # parser.add_argument('--graph_file', default='../data/new_vision_graph.pkl')#'../data/graphwithVN.txt')#('../data/graphwithVN.txt')#这里应该有一个n*3维的矩阵，但是俺没有
    
    # parser.add_argument('--model_path', default='../data/new_vision_graph.pkl')
    # parser.add_argument('--graph_file', default='graph.pkl')
    args = parser.parse_args()
    
    # +1 是对应着虚拟节点
    parser.add_argument('--num_nodes',type=int, default= 500)# 默认节点个数
    args = parser.parse_args()#args中存储了parser加入的变量
    
    # networkName = 'Karate_graph_transform/pkl'# 'embedding_result_line_SF'
    # nameList = ['../data/'+networkName+'/network1.pkl', '../data/'+networkName+'/network2.pkl',
    #                       '../data/'+networkName+'/network3.pkl', '../data/'+networkName+'/network4.pkl',
    #                       '../data/'+networkName+'/network5.pkl', '../data/'+networkName+'/network6.pkl',
    #                       '../data/'+networkName+'/network7.pkl', '../data/'+networkName+'/network8.pkl',
    #                       '../data/'+networkName+'/network9.pkl', '../data/'+networkName+'/network10.pkl']
    # 对于单张的图
    nameList = ['../data/来自line的数据/SF_N=500.pkl']
    # nameList = ['../data/化为数据/Facebook_Data.pkl', '../data/化为数据/Instagram_Data.pkl','../data/化为数据/Twitter_Data.pkl']
    
    if args.mode == 'train':
        for i,tar_g in enumerate(nameList):
            args.graph_file = tar_g
            tf.reset_default_graph()
            train(args,i)
    # elif args.mode == 'test':
    #     test(args)
def test(args):# 目前没有用
    pass
    # data_loader = DBLPDataLoader(graph_file=args.graph_file)
    # args.num_of_nodes = data_loader.num_of_nodes
    # model = LINEModel(args)
    # # 在这里读入训练好的模型
    # with tf.Session() as sess:
    #     print(args)
    #     print('batches\tloss\tsampling time\ttraining_time\tdatetime')
    #     tf.global_variables_initializer().run()
    #     initial_embedding = sess.run(model.embedding)
    #     learning_rate = args.learning_rate
    #     sampling_time, training_time = 0, 0
    #     t1 = time.time()
    #     u_i, u_j, label = data_loader.fetch_batch(batch_size=args.batch_size, K=args.K)
    #     feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}
    #     t2 = time.time()
    #     sampling_time += t2 - t1
    #     loss = sess.run(model.loss, feed_dict=feed_dict)
    #     print('t%f\t%0.2f\t%0.2f\t%s' % (loss, sampling_time, training_time,
    #                                         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    #     # 在这里使用训练好的模型进行embedding，然后保存到txt文件，但是不知道这个编码是不是对某个图专门进行训练编码的，保存模型没有意义？

def train(args,i):
    save_path = args.graph_file.split('/')[-1].split('.')[0]
    data_loader = DBLPDataLoader(graph_file=args.graph_file)
    suffix = args.proximity# string:second-order
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
                sess.run(model.train_op, feed_dict=feed_dict)#  sess.run()可以将tensor格式转成numpy格式
                training_time += time.time() - t2
                if learning_rate > args.learning_rate * 0.0001:
                    learning_rate = args.learning_rate * (1 - b / args.num_batches)
                else:
                    learning_rate = args.learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('%d\t%f\t%0.2f\t%0.2f\t%s' % (b, loss, sampling_time, training_time,
                                                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
                sampling_time, training_time = 0, 0
            # if b % 1000 == 0 or b == (args.num_batches - 1):
            #     embedding = sess.run(model.embedding)
            #     normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            #     pickle.dump(data_loader.embedding_mapping(normalized_embedding),
            #                 open('data/embedding_%s.txt' % suffix, 'wb'))
            if b == (args.num_batches - 1):
                embedding = sess.run(model.embedding)#最终的编码结果，每个节点有embedding_dim个特征
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)# 归一化
                # embedding_save_address = f'../data/Karate_graph_transform/embedding_line/_embedding{i}.txt'# 编码好的对象保存路径
                embedding_save_address = '../data/来自line的数据/'+save_path+'_line.txt'# 编码好的对象保存路径
                # embedding_save_address = '_embedding.txt'
                with open(embedding_save_address,'w') as f:
                    for i in range(args.num_nodes):
                        for j in range(args.embedding_dim):
                            f.write(str(normalized_embedding[i][j])+' ')
                        if i != (args.num_nodes-1):
                            f.write('\n')

if __name__ == '__main__':
    tf.reset_default_graph()# 因为上一次运行保存了变量，需要重置
    main()

# 不确定line中使用的那个确实的py文件使用的是什么方式将图转换为指定节点数的图，因此copy了一个128nodes的图进行embedding
# txtpath=r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\ERL-d1\code\data\graphwithVN.txt'
# pp=r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\line-master\data\_embedding.txt'

# # # graph_file = r'../data/co-authorship_graph.pkl'
# new_file_path = r'../data/new_vision_graph.pkl'

# DBLPDataLoader(new_file_path)

# with open(new_file_path, 'rb') as f:
#     G1 = pickle.load(f)
# nodes = list(G1.nodes)[:128]

# # 创建一个新的子图
# subgraph = G1.subgraph(nodes)
# #可视化
# nx.draw(G, with_labels=True)
# plt.show()

