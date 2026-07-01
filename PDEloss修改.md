# PDGCN PDE Loss 对流项修改说明

## 1. 修改目标

本次修改目标是将 PDGCN 的 PDE loss 对流项从原始的 ReLU 截断形式，改为**带符号边方向贡献方法**。

修改后的目标不是进行 FEM 弱形式矩阵装配，而是在图结构上更合理地近似 FEM 对流项中的核心物理量：

$$
\mathbf v_{scan}\cdot\nabla T
$$

---

## 2. FEM 中对流项的核心形式

在 FEM 中，对流项通常来自连续 PDE：

$$
\mathbf v_{scan}\cdot\nabla T
$$

对应弱形式为：

$$
\int_\Omega
w
\left(
\mathbf v_{scan}\cdot\nabla T
\right)
d\Omega
$$

其核心含义是：

> 速度向量 \(\mathbf v_{scan}\) 与温度梯度 \(\nabla T\) 做点积。

因此，PDGCN 的 PDE loss 对流项应尽量近似：

$$
\left(
\mathbf v_{scan}\cdot\nabla T
\right)_i
$$

---

## 3. 原始 PDGCN 对流项

原始对流项为：

$$
\left[\mathbf{v}_{scan}^*\cdot\nabla^*T^*\right]_i
\approx
\sum_{j\in\mathcal{N}(i)}
v_{scan}^*
\mathrm{ReLU}(\cos\theta_{ij})
\frac{T_i^*-T_j^*}{d_{ij}^*}
$$

其中：

- \(\mathcal N(i)\)：节点 \(i\) 的邻居集合；
- \(d_{ij}^*\)：节点 \(i\) 与节点 \(j\) 的无量纲距离；
- \(\theta_{ij}\)：扫描速度方向与边方向的夹角；
- \(\mathrm{ReLU}(\cos\theta_{ij})\)：只保留与扫描方向一致的邻居贡献。

该形式的问题是：

- 反方向邻居被直接截断；
- 对流方向由 ReLU 人为筛选；
- 与标准 \( \mathbf v_{scan}\cdot\nabla T \) 的带符号点积形式不完全一致。

---

## 4. 修改后：带符号边方向贡献方法

定义节点 \(i\) 到邻居节点 \(j\) 的边向量：

$$
\mathbf r_{ij}
=
\mathbf x_j-\mathbf x_i
$$

边长度：

$$
d_{ij}
=
\|\mathbf x_j-\mathbf x_i\|
$$

单位边方向：

$$
\mathbf e_{ij}
=
\frac{\mathbf x_j-\mathbf x_i}{d_{ij}}
$$

沿边方向的温度差分近似为：

$$
\frac{T_j-T_i}{d_{ij}}
$$

速度在该边方向上的投影为：

$$
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
$$

因此，边 \(i\to j\) 对对流项的贡献为：

$$
\left(
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
\right)
\frac{T_j-T_i}{d_{ij}}
$$

对所有邻居求和，得到修改后的 PDE loss 对流项：

$$
\left(
\mathbf v_{scan}\cdot\nabla T
\right)_i
\approx
\sum_{j\in\mathcal N(i)}
\alpha_{ij}
\left(
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
\right)
\frac{T_j-T_i}{d_{ij}}
$$

其中 \(\alpha_{ij}\) 是邻居权重。

---

## 5. 权重 \(\alpha_{ij}\) 的选择

使用距离归一化权重：

$$
\alpha_{ij}
=
\frac{d_{ij}^{-p}}
{\sum_{k\in\mathcal N(i)} d_{ik}^{-p}}
$$

其中 \(p\) 取 \(1\) 。

---

## 6. 与原始 ReLU 形式的关键区别

原始方法使用：

$$
\mathrm{ReLU}(\cos\theta_{ij})
$$

因此：

$$
\cos\theta_{ij}<0
\quad\Rightarrow\quad
\mathrm{ReLU}(\cos\theta_{ij})=0
$$

反方向邻居不会参与对流项计算。

修改后使用：

$$
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
$$

该项保留正负号：

- 若 \(\mathbf v_{scan,i}\cdot\mathbf e_{ij}>0\)，说明边方向与速度方向大致一致；
- 若 \(\mathbf v_{scan,i}\cdot\mathbf e_{ij}<0\)，说明边方向与速度方向相反；
- 若 \(\mathbf v_{scan,i}\cdot\mathbf e_{ij}=0\)，说明边方向与速度方向近似垂直。

因此，修改后的方法不再截断反方向邻居，而是保留其带符号贡献。

---

## 7. 修改前后公式对比

### 修改前

$$
\left[\mathbf{v}_{scan}^*\cdot\nabla^*T^*\right]_i
\approx
\sum_{j\in\mathcal{N}(i)}
v_{scan}^*
\mathrm{ReLU}(\cos\theta_{ij})
\frac{T_i^*-T_j^*}{d_{ij}^*}
$$

特点：

- 使用 \(\mathrm{ReLU}(\cos\theta_{ij})\)；
- 只保留与扫描方向一致的邻居；
- 反方向贡献被截断。

### 修改后

$$
\left(
\mathbf v_{scan}\cdot\nabla T
\right)_i
\approx
\sum_{j\in\mathcal N(i)}
\alpha_{ij}
\left(
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
\right)
\frac{T_j-T_i}{d_{ij}}
$$

特点：

- 不使用 \(\mathrm{ReLU}(\cos\theta)\)；
- 使用带符号方向投影 \(\mathbf v_{scan,i}\cdot\mathbf e_{ij}\)；
- 保留上下游邻居的正负贡献；
- 更接近 \( \mathbf v_{scan}\cdot\nabla T \) 的点积形式；
- 计算开销接近原始图边差分方法。

---

## 8. PDE loss 中的残差写法

若 PDE 残差中需要对流项，可将对流残差写为：

$$
R_{conv,i}
=
\sum_{j\in\mathcal N(i)}
\alpha_{ij}
\left(
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
\right)
\frac{T_j-T_i}{d_{ij}}
$$

然后将其代入整体 PDE 残差：

$$
R_i
=
R_{time,i}
+R_{conv,i}
+R_{diff,i}
-R_{source,i}
$$

最终 PDE loss 可写为：

$$
\mathcal L_{PDE}
=
\frac{1}{N}
\sum_{i=1}^{N}
R_i^2
$$

---

## 9. 结论

本次 PDE loss 对流项固定修改为带符号边方向贡献方法：

$$
R_{conv,i}
=
\sum_{j\in\mathcal N(i)}
\alpha_{ij}
\left(
\mathbf v_{scan,i}\cdot\mathbf e_{ij}
\right)
\frac{T_j-T_i}{d_{ij}}
$$

该方法的核心变化是：

- 删除 \(\mathrm{ReLU}(\cos\theta)\)；
- 不再截断反方向邻居；
- 使用 \(\mathbf v_{scan,i}\cdot\mathbf e_{ij}\) 表示速度在边方向上的带符号投影；
- 用边方向温度差分 \((T_j-T_i)/d_{ij}\) 近似方向导数；
- 在图结构上近似 FEM 对流项的核心形式 \( \mathbf v_{scan}\cdot\nabla T \)。
