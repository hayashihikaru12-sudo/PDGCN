# 参考资料说明：

本代码库的目的是：构建曲面内PDGCN神经网络训练架构及曲面差分技术，实现对多层堆叠的铺放曲面温度场预测，PDGCN架构改造自PIGNN架构，面向曲面铺放热场预测应用场景

1. @/DesignPlan文件夹中，具体说明了PDGCN+FDM温度场预测技术发难、初温处理方案、无量纲化处理方案及损失函数构建策略
2. @/DesignPlan/1.h5 是预先生成的输入数据（用于训练）
3. @/PIGNN文件夹中：为PIGNN仓库源码，具体展示了PIGNN架构的构建、训练及推理的全过程

本代码库的python环境是：**conda中的PIGNN环境**（D:\ProgramData\\CondaEnv\PIGNN\python.exe），已经配置好了

**@/DesignPlan文件夹和@/PIGNN文件夹均为参考资料文件夹，均可读取不可修改**

