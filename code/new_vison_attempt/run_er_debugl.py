import numpy as np, os, time, random
import torch
import replay_memory

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from influence_v2 import Env
from Agent import Agent

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def getResultPath(gra_p, mood):# 创建文件夹保存结果
    path = "result/"+gra_p[0].split('/')[-1].split('.')[0]+mood
    if os.path.exists(path):
        return path
    os.makedirs(path)# 创建文件夹
    # os.makedirs(path + "/embedding/")
    os.makedirs(path + "/model/")
    return path# 每次调用都只在result文件夹下创建一个子文件夹
    # for i in range(1,21):
    #     path = "result/" + str(i) + "/"
    #     if os.path.exists(path):
    #         continue
    #     os.makedirs(path)# 创建文件夹
    #     # os.makedirs(path + "/embedding/")
    #     os.makedirs(path + "/model/")
    #     return path# 每次调用都只在result文件夹下创建一个子文件夹


if __name__ == "__main__":
    # Create Env
    t1 = time.time()
    # gra_p = ['../data/生成数据/BA_N=200_k=4_edgelist','../data/生成数据/ER_N=200_k=4_edgelist','../data/生成数据/Fried_edgelist_224']
    # embed_p = ['../data/生成数据/BA_N=200_k=4_line_1.txt','../data/生成数据/ER_N=200_k=4_edgelist','../data/生成数据/Fried_edgelist_224_line_1.txt']
    # removed_p = ['../data/生成数据/BA_N=200_k=4_edgelist_removed.txt','../data/生成数据/ER_N=200_k=4_edgelist_removed.txt','../data/生成数据/Fried_edgelist_224_removed.txt']
    
    gra_p = ['../data/来自line的数据/SF_N=500']
    
    # embed_p = ['../data/来自line的数据/Emb_SF_N=500_0']
    # embed_p = ['../data/来自line的数据/SF_N=500_line_2dim.txt']
    embed_p = ['../data/来自line的数据/Emb_SF_N=500_0']
    # embed_p = [r'E:\coursewares\SchoolCourses\大三了唉下\人工智能技术驱动的网络信息挖掘\230221_一些初步资料\sden编码模型\SDNE-master-fromgit\result\WS_N_5_layer_net-Fri_1422_4955\WS_N_5_layer_net_sdne.txt']
    # embed_p = ['../data/生成数据2/WS_2_layer_sdne.txt']
    
    removed_p = ['../data/来自line的数据/WS_N=200_k=4_edgelist_removed.txt']
    
    # gra_p = ['../data/化为数据/Facebook_Data.txt_edgelist', '../data/化为数据/Instagram_Data.txt_edgelist','../data/化为数据/Twitter_Data.txt_edgelist']
    # embed_p = ['../data/化为数据/Facebook_Data_line.txt', '../data/化为数据/Instagram_Data_line.txt','../data/化为数据/Twitter_Data_line.txt']
    # mood = '_line'
    embed_dim = 2
    epoch = 80# 进化迭代次数
    mood = f'_sdne_sdnedata_{embed_dim}_dim_round{epoch}'
    mainPath = getResultPath(gra_p, mood)# 创建子文件夹用于存储结果的模型和图

    env = Env(mainPath, gra_p, embed_p, embed_dim)# mainPath用于传递存储result的路径
    # Random seed setting随机种子
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #Create Agent
    agent = Agent(env)

    next_save = 100; time_start = time.time()
    print("Start training")
    maxList = np.array([])
    resultList = np.array([])
    timeList = np.array([])
    
    all_fit_change = []
    all_seeds_change = []
    attack_contrast = []
    seeds_contrast = []
    averages = []
    
    for graphIndex in range(len(agent.env.nameList)):# 遍历每一个含图的文件，对每一个图文件依次得到影响、时间、种子集
        fit_change = []
        seeds_change = []
        for i in range(epoch):# Generation，种群迭代轮数，单独运行这个循环以继续训练
            if i == 0:
                agent.evaluate_all()
                # 对于agent中种群的每个个体网络DQN网络，按其Q值选出一个种子节点集，将返回的total_reward加入到list成员all_fitness
                # 每个DQN找一次种子集，但是env中只保留最后一次求出的种子集
            best_train_fitness, average, rl_agent, elitePop = agent.train()
            averages.append(average)
            if i % 10==0:
                print("=================================================================================")
                print(graphIndex, "th graph      Generation:", i, "    ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                # rl_agent是我们训练的目标DQN，是pop种群中最佳的一个参数
                print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness, ' Avg:', average)
            maxList = np.append(maxList, best_train_fitness)
            # 最优解变化情况
            fitness, seeds = agent.evaluate(agent.pop[0])
            fit_change.append(fitness)
            seeds_change.append(seeds)

            # #Save Policy
            # torch.save(rl_agent, mainPath + "//model//rlmodel" + str(graphIndex))# 存储训练的DQN参数
            # for eliteIndex in range(len(elitePop)):# elitePop是pop种群中前10%的个体
            #     torch.save(elitePop[eliteIndex], mainPath + "//model//elite_model" + str(graphIndex) + "_" + str(eliteIndex))
            # np.savetxt(mainPath + "//maxList.txt", maxList)
            # print("Progress Saved")
            # if i == epoch-1:
            #     # 使用当前DQN模型对破坏前后的影响力进行评估对比
            #     agent.env.setGraph(gra_p[graphIndex], embed_p[graphIndex])
            #     fitness_or, seeds_or = agent.evaluate(agent.pop[0])
                
                # 破坏评估
                # agent.env.setGraph(removed_p[graphIndex], embed_p[graphIndex])
                # fitness_bro, seeds_bro = agent.evaluate(agent.pop[0])
                
                # attack_contrast.append([fitness_or,fitness_bro])
                # seeds_contrast.append(seeds_or)
                # seeds_contrast.append(seeds_bro)
                
                # print(f'被破坏前影响力大小为：{fitness_or},移除边影响力下降为：{fitness_bro}')
        
        all_fit_change.append(fit_change)
        all_seeds_change.append(seeds_change)
        
        fitness, seeds = agent.evaluate(agent.pop[0])
        # pop[0]是当前影响力最大的DQN网络，每次对一个图进行训练之后使用获得的best选出种子，
        print("best fitness:", fitness)
        print("seeds:", seeds)
        resultList = np.append(resultList, fitness)
        t2 = time.time()
        timeList = np.append(timeList, t2-t1)# 存储每一张图用于训练的时长
        t1 = t2
        np.savetxt(mainPath + "//seeds_" + embed_p[graphIndex].split('/')[-1], seeds, fmt="%d")
        np.savetxt(mainPath + "//averages" , averages, fmt="%.8f")
        if graphIndex < len(agent.env.nameList) - 1:
            agent.env.nextGraph()# 如果当前不是训练list中的最后一张图，则调用next设置为下一张txt存储的图
            agent.replay_buffer = replay_memory.ReplayMemory(agent.buffer_size)
        else:
            break

    # 画出变化情况
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # 设置画布和子图
    fig, axs = plt.subplots(2, 5, figsize=(10, 8), sharey=True)
    
    # 绘制子图
    for i in range(len(all_fit_change)):
        sns.lineplot(x=range(len(all_fit_change[i])), y=all_fit_change[i], ax=axs[int(i/5)][int(i%5)])
        axs[int(i/5)][int(i%5)].set_xlabel('Index')
        axs[int(i/5)][int(i%5)].set_ylabel('Data')
    
    # 设置标题和美化
    fig.suptitle('Data Visualization', fontsize=20)
    sns.set(style="whitegrid")
    sns.despine()
    plt.tight_layout()
    plt.show()


    print("time cost:")
    agent.showScore(timeList)
    # 使用前面几张图作为训练DQN的数据，然后用最后一张图训练+求出种子
    # 这样需要区分list中的种子是不是代表一个或者一类图吗？，这个list中的图有什么关联，
    # what if list中只给出一张图
    print("influence:")
    agent.showScore(resultList)
    np.savetxt(mainPath + "//timeList.txt", timeList)# 每张图上花费的时间
    np.savetxt(mainPath + "//resultList.txt", resultList, fmt='%.8f')
    # 在每张图上的结果---则results中存储的都不是一张图上的结果变化，而是模型对于每张图都进行一定次数的训练后得到的结果
    # 那么久有可能说这个list里是为了看一个平均值，即模型对多张图片的效果，而不是将前面的图片作为训练数据
    # np.savetxt(mainPath + "//seeds.txt", seeds, fmt="%d")# 最后一张图的种子
    # np.savetxt(mainPath + "//删除前后对比.txt", attack_contrast, fmt='%.8f')
    # np.savetxt(mainPath + "//删除前后重选种子对比.txt", seeds_contrast, fmt='%d')


# # """查看pop中所有种群的结果"""
# a_s2 = []
# a_inf2 = []
# for k in range(len(agent.pop)):
#     fitness, seeds = agent.evaluate(agent.pop[k])
#     a_inf2.append(fitness)
#     a_s2.append(seeds)
# print(max(a_inf2))

