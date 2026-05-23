# 参考资料说明：

本代码库的目的是：构建曲面内PDGCN神经网络训练架构及曲面差分技术，实现对多层堆叠的铺放曲面温度场预测，PDGCN架构改造自PIGNN架构，面向曲面铺放热场预测应用场景

1. @/DesignPlan文件夹中，具体说明了研究背景、PDGCN+FDM温度场预测技术方案、初温处理方案、无量纲化处理方案及损失函数构建策略
2. @/DesignPlan/局部窗口定拓扑采样器.md：文件说明了单层定拓扑计算域的构建过程

本代码库的python环境是：**conda中的PIGNN环境**（D:\ProgramData\\CondaEnv\PIGNN\python.exe），已经配置好了

**@/DesignPlan文件夹和@/PIGNN文件夹均为参考资料文件夹，均可读取不可修改（除非prompt显式要求）**

