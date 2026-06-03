# 项目说明

本代码库用于构建曲面内 PDGCN 神经网络训练架构和曲面差分技术，实现多层堆叠铺放曲面温度场预测。PDGCN 架构改造自 PIGNN，面向曲面铺放热场预测应用场景。

## 当前架构

当前代码采用显式算子分裂：

```text
T_next = T_current + delta_T_source + delta_T_inplane + delta_T_thickness
```

- 显式表面热源模块读取 HDF5 `dynamic/Q`，将表面热流 `q''` 转为顶层温升。
- PD-GCN 是无源曲面内输运算子，节点输入为 `[x*, y*, z*, fx, fy, fz, T*]`。
- PDE residual 是无源输运 residual，不再包含热源项或单层等效热汇项。
- 厚度方向 1D FDM 只负责层间导热；底层在单步更新后钳制为指定底部温度。

## 参考资料

1. `DesignPlan/` 中说明研究背景、PDGCN+FDM 温度场预测方案、初温处理、无量纲化和损失函数策略。
2. `DesignPlan/局部窗口定拓扑采样器.md` 说明单层定拓扑计算域构建过程。
3. `PIGNN/` 为 PIGNN 参考仓库源码。

除非 prompt 显式要求，`DesignPlan/` 和 `PIGNN/` 均作为参考资料目录处理。本次改造已显式要求同步更新 `DesignPlan/`。

## Python 环境

本代码库的 Python 环境是 conda 中的 `PIGNN` 环境：

```powershell
D:\ProgramData\CondaEnv\PIGNN\python.exe
```
