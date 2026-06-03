# 项目说明

本代码库用于构建曲面内 PDGCN 神经网络训练架构和曲面差分技术，实现多层堆叠铺放曲面温度场预测。PDGCN 架构改造自 PIGNN，面向曲面铺放热场预测应用场景。

## 当前物理职责划分

总体更新式为：

```text
T_next = T_current + delta_T_source + delta_T_inplane + delta_T_thickness
```

- `delta_T_source`：由显式表面热源模块计算，只默认作用于顶层。
- `delta_T_inplane`：由无源 PD-GCN 计算，只负责曲面内对流、曲面扩散和纤维各向异性扩散。
- `delta_T_thickness`：由厚度方向 1D 显式 FDM 计算，负责层间导热和底层恒温边界。

PD-GCN 节点输入固定为：

```text
[x*, y*, z*, fx, fy, fz, T*]
```

热源 `dynamic/Q` 不进入节点特征；它按表面热流 `q''` 读取，转换为 `W/m^2` 后由显式表面热源模块处理。

## 参考资料

1. `DesignPlan/` 中说明研究背景、PDGCN+FDM 温度场预测方案、初温处理、无量纲化和损失函数策略。
2. `DesignPlan/局部窗口定拓扑采样器.md` 说明单层定拓扑计算域构建过程。

本次架构改造已要求同步修改 `DesignPlan/`。后续若没有显式要求，仍将 `DesignPlan/` 和 `PIGNN/` 作为参考资料目录，避免随意改动。

## Python 环境

本代码库的 Python 环境是 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```
