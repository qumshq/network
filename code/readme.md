## 文件结构

==new_vison_attempt==文件夹改写后的模型

DQN_learning是原来的模型一起传上去了

|-- new_vison_attempt
|   |-- DQN_Model.py
|   |-- SSNE.py
|   |-- \_\_pycache\__
|   |-- degree.py 度，好像在模型中没有用到
|   |-- influence1.py
|   |-- line.py
|   |-- line_model.py
|   |-- line_utils.py
|   |-- mod_utils.py
|   |-- new_graph.pkl
|   |-- repeat.py
|   |-- replay_memory.py
|   |-- run_er_debugl.py

## line模型

|   |-- line.py
|   |-- line_model.py
|   |-- line_utils.py

line.py是主要文件

​		读取pickle文件中存储的图，然后进行编码最终对每个节点得到embedding_dim(默认为128)个特征；最终得到一个n*embedding_dim的矩阵，将其存储在_embedding.txt文件中。

​		==这里使用的pickle图文件：使用node edge表示节点和边(无权图)，然后存储在pickle文件中==

## run_er_debugl.py

调用其他所有文件，将data中的图使用line进行编码，然后训练DQN网络选出种子节点并保存结果