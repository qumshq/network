# 一些需要的库的验证
import matlab.engine
import scipy.io as scio
import numpy as np
import os
import networkx as nx
import torch
import random


a = torch.zeros((3, 4))
ind_cr = random.randint(0, a.shape[0])
a[ind_cr, :] = a[ind_cr, :]

print(a)


# eng = matlab.engine.start_matlab()
# # localpath = "D://PythonProjects//deep-q-learning-master//code//"
# localpath = r"E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\ERL-d1\code\data"
# embedPath = localpath + 'data//val.npy'
# embedding = np.load(embedPath)

# embeddingOrder = [0 for _ in range(len(embedding))]


# with open(localpath + 'data//val.txt', "r") as f:
#     for i in range(len(embedding)):
#         data = f.readline()
#         index = int(data)
#         embeddingOrder[index] = embedding[i]

# if (embeddingOrder[274] == embedding[0]).all():
#     print("right!")



# dataFile = localpath + 'data//embedding.mat'
# data = scio.savemat(dataFile, {'embed': embeddingOrder})

# print(embedding.shape)

# eng.embed2kmeans()
