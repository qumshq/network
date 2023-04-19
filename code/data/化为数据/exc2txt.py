# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 15:11:54 2023

@author: Henry
"""
import pandas as pd
import numpy as np

# # 读取Excel文件
# # df_contrast = pd.read_excel('Facebook_Data.xlsx')
# paths = ['Facebook_Data.xlsx','Instagram_Data.xlsx','Twitter_Data.xlsx']
# for fi in paths:
#     df = pd.read_excel(fi, header=None)
#     df = df.drop(df.columns[0], axis=1)
#     df = df.drop(0, axis=0)
#     df = np.array(df)
#     save_p = fi.split('.')[0]
#     np.savetxt(save_p+'.txt', df, fmt='%d', delimiter='')

# 对于line编码，需在得到哦邻接表后在第一行插入所有行数+边数的一行
nameList = ['Facebook_Data.txt_edgelist', 'Instagram_Data.txt_edgelist','Twitter_Data.txt_edgelist']
for n in nameList:
    g = np.loadtxt(n)
    add = [np.max(g)+1,len(g)]
    print(add)
    g_n = np.row_stack((add,g))
    np.savetxt(n.split('.')[0] + '_stastics.txt', g_n, fmt='%d')
