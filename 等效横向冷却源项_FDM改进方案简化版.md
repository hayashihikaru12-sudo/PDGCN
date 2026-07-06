# 瞬态散热片模型：厚度方向传热改进方案（简化版）

## 1. 问题

当前 1D 隐式 FDM 在下层缺失面内热扩散，温度剖面近乎线性，中层峰值被高估 100–250 °C。

## 2. 控制方程

将厚度传热替换为**瞬态散热片模型**：

$$
\boxed{
\frac{\partial T^*}{\partial t^*}
= k_{\text{ratio}} \cdot \text{Pe}^{-1} \cdot \frac{\partial^2 T^*}{\partial z^{*2}}
- \gamma^* \cdot \bigl(T^* - T^*_{\text{bottom}}\bigr)
}
$$

- 左端：瞬态项
- 右端第一项：厚度方向导热（即原 FDM）
- 右端第二项：**新增**——等效横向冷却，将缺失的面内扩散建模为向模具（$T_{\text{bottom}}$ = 120 °C）的散热

## 3. 参数

唯一新增参数 $\gamma^*$ 完全由现有代码参数表达，无需 FEM 标定：

$$
\boxed{
\gamma^* = \frac{\text{Pe}^{-1}}{R_{\text{char}}^2},
\qquad
\gamma^* \Delta t^* = \frac{\text{Pe}^{-1} \cdot \Delta t^*}{R_{\text{char}}^2}
}
$$

剖面弯曲度由无量纲参数 $\beta H$ 控制：

$$
\boxed{
\beta^* H^* = \frac{H^*}{R_{\text{char}} \sqrt{k_{\text{ratio}}}}
}
$$

| 参数 | 含义 | 来源 |
|------|------|------|
| $\text{Pe}^{-1}$ | 逆 Peclet 数 | 配置 `inverse_pe` |
| $k_{\text{ratio}}$ | 导热各向异性比 | 配置 |
| $\Delta t^*, \Delta z^*, H^*$ | 时间步长、层间距、总厚度 | 配置 |
| $R_{\text{char}}$ | 热源特征半径 / $L_0$ | HDF5 `dynamic/Q` 空间 FWHM 估算，默认 0.1 |

**全参数零 FEM 标定。**

## 4. 数值实现

Backward Euler 隐式格式，仅修改三对角矩阵对角线：

$$
\mathbf{A}_{\text{fin}} = \mathbf{I} - C_n \mathbf{D} + \gamma^* \Delta t^* \cdot \mathbf{I}
$$

右端项和底层 Dirichlet 边界不变。改动仅 `pde/fdm.py` ~20 行，`gamma=None` 时退化为现有 FDM。

## 5. 目标

中层 T_peak 偏差从 150–250 °C 降至 50–110 °C，剖面由线性转为指数衰减。
