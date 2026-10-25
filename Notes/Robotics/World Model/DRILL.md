---
title: "DRILL: Training World Models to Improve Policies, Not Predict Pixels"
method_name: "DRILL"
authors: [Zhaolu Kang, Tailong Luo, Lei Wei, Guangyuan Dong, Sihan Liu, Yingjie He, Siheng Wang, Shuaibo Li, Rongchao Zhang, Zhen Tian, Richeng Xuan, Zhichao Hu]
year: 2026
venue: NeurIPS 2026
tags: [world-model, reinforcement-learning, vla, bilevel-optimization, policy-optimization, meta-gradient]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.30814v1
created: 2026-10-24
---

# 论文笔记：DRILL: Training World Models to Improve Policies, Not Predict Pixels

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | 未列出（arXiv 版本）|
| 日期 | October 2026 |
| 项目主页 | 未公开 |
| 对比基线 | [[WMPO]]、[[VLAW]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.30814) |

---

## 一句话总结

> DRILL 通过[[双层优化]]将[[世界模型]]训练目标从像素 MSE 替换为"策略在虚拟 rollout 后的真实环境回报"，消除像素保真度与策略提升之间的目标错配，在五个仿真器上平均成功率比 WMPO 高 13.1 个百分点。

---

## 核心贡献

1. **目标错配诊断**: 证明更低的像素 MSE 可以产生更差的策略梯度——17% 更高 MSE 的世界模型反而能带来更好的下游策略
2. **DRILL 双层框架**: 提出两个变体：[[DRILL-VWI]]（无需额外仿真器交互的闭式代理）和 [[DRILL-IMG]]（通过[[截断BPTT]]精确估计元梯度），通过 Proposition 1 建立两者理论关联
3. **收敛性保证**: Theorem 1 给出截断 BPTT 下非渐近收敛速率，包含优化误差、方差项和截断偏差三项
4. **PUR 诊断工具**: 提出 [[Policy-Utility Residual]]（PUR）度量像素 MSE 梯度与策略回报梯度的对齐程度，Spearman ρ = −0.89 与策略效用高度相关

---

## 问题背景

### 要解决的问题

当前[[视觉-语言-动作模型|VLA]] 的强化学习后训练（post-training）流程依赖[[世界模型]]提供虚拟 rollout 以生成策略梯度信号。现有[[世界模型]]以**像素预测**为训练目标（最小化下一帧像素 MSE），但其真实用途是**为策略提供有用的学习信号**。这两个目标存在根本性错配：预测精确的像素并不等价于生成有利于策略提升的状态转移分布。

### 现有方法的局限

- **[[WMPO]]（World Model Policy Optimization）**: 标准双阶段流程——先用 MSE 拟合[[世界模型]]，再在其中运行 [[GRPO]] 更新策略；[[世界模型]]完全以像素重建为目标，与策略优化解耦
- **[[VLAW]]**: 在 WMPO 基础上加入额外正则化，但仍沿用像素 MSE 目标
- **[[DreamerV3]]、[[MuZero]]**: 基于价值感知模型学习（VAML）等方法更新[[世界模型]]，但缺乏面向 VLA 后训练场景的理论保证和实证验证
- **根本缺陷**: 像素 MSE 对所有像素等权重，而策略梯度信号只依赖任务相关区域的预测质量

### 本文的动机

若直接将[[世界模型]]的训练信号替换为"策略在其中学习后的实际表现"，则两者目标天然对齐。DRILL 通过[[双层优化]]实现：外层优化[[世界模型]]参数 $\phi$，使内层经 [[GRPO]] 更新后的策略 $\pi_{\theta^+(\phi)}$ 在**目标环境**中获得最高回报。

---

## 方法详解

### 模型架构

DRILL 在 [[WMPO]] 的三组件基础上修改[[世界模型]]训练目标：

- **输入**: 演示数据 $\mathcal{D}_{\mathrm{demo}}$（状态-动作-下一状态三元组）+ 少量真实仿真器交互（DRILL-IMG 专用）
- **[[世界模型]] $\mathcal{W}_\phi$**: 视频预测模型（Gaussian head），预测 $\hat{o}_{t+1}(\phi)$
- **策略 $\pi_\theta$**: VLA 骨干（[[π0]]-3B 或 [[OpenVLA]]-7B）
- **冻结奖励模型 $r$**: 预训练 VLM，提供任务完成奖励
- **总参数**: π0-3B 骨干（3B）+ 世界模型（300M / 1.2B / 7B 三种规模）

### 核心模块

#### 模块1: DRILL-VWI（Visitation-Weighted Imagination）

**设计动机**: 利用[[双层优化]]的一阶近似，无需额外仿真器 rollout 即可近似元梯度

**具体实现**:
- 采样演示锚点，在[[世界模型]]下展开 $H$ 步虚拟 rollout
- 使用 [[GRPO]] 计算 Group-Relative 优势 $\hat{A}(o_t, a_t)$
- 通过 kNN 核密度估计（隐空间 buffer）估计[[策略访问分布]] $\tilde{d}_\phi^{\pi_\theta}(o_t)$
- 构建三项复合权重 $w = \text{访问密度} \times \text{符号优势幂次} \times \text{世界模型置信度}$
- 对加权 MSE 做梯度下降更新 $\phi$；不产生任何新的仿真器交互

#### 模块2: DRILL-IMG（Implicit Meta-Gradient）

**设计动机**: 通过内层 [[GRPO]] 更新的[[截断BPTT]]精确估计双层目标的元梯度

**具体实现**:
- 在[[世界模型]]中展开虚拟 rollout，执行一步内层策略更新 $\theta^+ \leftarrow \theta + \eta_\theta \nabla_\theta \hat{J}_{\mathrm{GRPO}}$
- 在真实仿真器中收集 $B_{\mathrm{real}} = 30$ 条轨迹，得到外层策略梯度 $G$
- 计算 $K=2$ 步截断混合 Hessian-向量积（HVP）：$G \cdot \nabla^2_{\theta,\phi} \hat{J}_{\mathrm{GRPO}}$，通过[[反向传播]]实现
- 更新 $\phi$ 以最大化 $J(\pi_{\theta^+(\phi)})$

---

## 关键公式

### 公式1: [[像素MSE损失|WMPO 世界模型目标]]

$$
\mathcal{L}_{\mathrm{MSE}}(\phi) = \mathbb{E}_{(o_{\leq t},\,a_t,\,o_{t+1})\sim\mathcal{D}_{\mathrm{demo}}}\!\left[\bigl\|\hat{o}_{t+1}(\phi) - o_{t+1}\bigr\|_2^2\right]
$$

**含义**: WMPO 基线用演示数据的像素均方误差训练世界模型，是 DRILL 要替换的目标

**符号说明**:
- $\phi$: 世界模型参数
- $\hat{o}_{t+1}(\phi)$: 世界模型预测的下一帧
- $o_{t+1}$: 真实下一帧
- $\mathcal{D}_{\mathrm{demo}}$: 演示数据集

### 公式2: [[GRPO|WMPO 策略步]]

$$
\hat{J}_{\mathrm{GRPO}}(\pi_\theta;\mathcal{W}_\phi)=\mathbb{E}_{\tau\sim P_{\mathcal{W}_\phi}(\pi_\theta)}\!\left[\frac{1}{H}\sum_{t=0}^{H-1}\hat{A}(o_t,a_t)\log\pi_\theta(a_t\mid o_{\le t})\right]
$$

**含义**: 策略在世界模型内展开的虚拟轨迹上执行 GRPO 梯度更新

**符号说明**:
- $\theta$: 策略参数
- $\tau$: 在世界模型 $\mathcal{W}_\phi$ 下展开的轨迹
- $H$: rollout 时间步长（horizon）
- $\hat{A}(o_t, a_t)$: Group-Relative 优势估计

### 公式3: [[双层优化|内层更新后的策略]]

$$
\theta^+(\phi) = \theta + \eta_\theta \nabla_\theta \hat{J}_{\mathrm{GRPO}}(\pi_\theta;\mathcal{W}_\phi)
$$

**含义**: 在当前世界模型 $\mathcal{W}_\phi$ 中执行一步 GRPO 更新后的策略参数，是双层目标的内层解

**符号说明**:
- $\eta_\theta$: 策略学习率
- $\theta^+(\phi)$: 世界模型参数为 $\phi$ 时内层更新后的策略参数

### 公式4: [[双层优化|DRILL 双层目标]]

$$
\max_\phi \; J\!\left(\pi_{\theta^+(\phi)}\right), \quad J(\pi) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{H-1} r(o_t, a_t)\right]
$$

**含义**: DRILL 的核心目标：选择世界模型参数 $\phi$，使得策略经内层 GRPO 更新后在**真实目标环境**中获得最高回报

**符号说明**:
- $J(\pi)$: 策略 $\pi$ 在目标环境中的期望累积奖励
- $r(o_t, a_t)$: 冻结 VLM 奖励模型给出的任务完成奖励
- $\phi$: 待优化的世界模型参数

### 公式5: [[DRILL-VWI|VWI 加权 MSE 损失]]

$$
\mathcal{L}_{\mathrm{VWI}}(\phi) = -\mathbb{E}_{(o_t,a_t,o_{t+1})\sim\mathcal{D}_{\mathrm{demo}}}\!\left[\bar{w}(o_t,a_t)\cdot\log\mathcal{W}_\phi(o_{t+1}\mid o_{\leq t},a_t)\right]
$$

**含义**: 将标准 MSE 替换为访问密度加权的对数似然损失，相当于对任务相关转移样本施加更大梯度权重

**符号说明**:
- $\bar{w}(o_t, a_t)$: 经 clip + 归一化的复合权重（见公式6）
- $\log\mathcal{W}_\phi(o_{t+1}\mid o_{\leq t},a_t)$: Gaussian 世界模型的对数似然，等价于负 MSE

### 公式6: [[DRILL-VWI|VWI 复合权重]]

$$
w(o_t,a_t) = \underbrace{\tilde{d}_\phi^{\pi_\theta}(o_t)}_{\text{访问密度}} \cdot \underbrace{\mathrm{sgn}(\hat{A})\cdot|\hat{A}|^\alpha}_{\tilde{A}_\alpha\text{（符号优势幂次）}} \cdot \underbrace{s(o_t)}_{\text{世界模型置信度}}
$$

**含义**: 三项权重共同决定每个演示转移在世界模型梯度更新中的贡献：密集访问的、优势符号正确的、世界模型置信度高的转移贡献更大

**符号说明**:
- $\tilde{d}_\phi^{\pi_\theta}(o_t)$: 策略在世界模型中的访问频率，用 kNN 核密度在隐空间 buffer 中估计
- $\alpha$: 优势幂次指数（默认 $\alpha=1$，消融显示 $\alpha=2$ 降至 67.9 vs 69.2）
- $s(o_t)$: 世界模型对当前观测的预测置信度（低置信度时降低权重）

### 公式7: [[Gaussian世界模型|Gaussian head 对数似然梯度]]

$$
\nabla_\phi\log\mathcal{W}_\phi(o_{t+1}\mid o_{\le t},a_t)=-\frac{1}{2\sigma_W^2}\nabla_\phi\bigl\|\hat{o}_{t+1}(\phi)-o_{t+1}\bigr\|_2^2
$$

**含义**: Gaussian observation head 下世界模型的得分函数（score function），将对数似然梯度等价为负 MSE 梯度；DRILL-VWI 和 DRILL-IMG 都借助此等价性连接世界模型梯度与策略改进梯度

**符号说明**:
- $\sigma_W^2$: Gaussian 输出头的固定方差
- $\hat{o}_{t+1}(\phi)$: 世界模型的确定性输出（均值预测）

### 公式8: [[元梯度|DRILL-IMG 元梯度展开]]

$$
\nabla_\phi J(\pi_{\theta^+(\phi)})=\eta_\theta\,\mathbb{E}_{\mu_{\mathrm{roll}}}\!\left[\hat{A}(o_t,a_t)\,q_\theta(o_t,a_t)\,\nabla_\phi\log\mathcal{W}_\phi(o_{t+1}\mid o_{\le t},a_t)\right]+\eta_\theta r_H+\mathcal{O}(\eta_\theta^2)
$$

**含义**: 将双层目标对 $\phi$ 的梯度展开为一阶近似，揭示元梯度与世界模型得分函数、GRPO 优势及策略兼容性分数 $q_\theta$ 的三方联系

**符号说明**:
- $q_\theta(o_t,a_t) = \langle\nabla_\theta\log\pi_\theta(a_t\mid o_{\leq t}),\,\nabla_\theta J(\pi_\theta)\rangle$: 策略得分-策略改进的内积
- $\mu_{\mathrm{roll}}$: 虚拟 rollout 的状态-动作-下一状态边缘分布
- $r_H$: 轨迹局部性残差
- $\mathcal{O}(\eta_\theta^2)$: 高阶项

### 公式9: [[DRILL-VWI|Proposition 1 一阶近似界]]

$$
\Bigl\|\nabla_\phi J(\pi_{\theta^+(\phi)})+\frac{\kappa\eta_\theta}{2\sigma_W^2}\nabla_\phi\mathcal{L}_{\mathrm{VWI}}^{\mathrm{core}}(\phi)\Bigr\|\le\eta_\theta M_W\epsilon_{\mathrm{comp}}+\eta_\theta R_H+\eta_\theta R_{\mathrm{demo}}+\mathcal{O}(\eta_\theta^2)
$$

**含义**: Proposition 1 核心结论：DRILL-VWI 的梯度方向近似于 DRILL-IMG 的元梯度，残差由三项控制——得分兼容性残差 $\epsilon_{\mathrm{comp}}$、轨迹局部性残差 $R_H$、演示锚点残差 $R_{\mathrm{demo}}$

**符号说明**:
- $\kappa$: 策略得分兼容性系数
- $M_W$: 世界模型 Lipschitz 常数
- $\epsilon_{\mathrm{comp}}$: 得分兼容性残差（$\epsilon_{\mathrm{comp}} \to 0$ 当策略得分与改进梯度对齐时）
- $R_H$: 轨迹局部性残差（rollout 时间步有限时产生）
- $R_{\mathrm{demo}}$: 演示锚点残差（虚拟 rollout 分布偏离演示分布时产生）

### 公式10: [[截断BPTT|Theorem 1 收敛界]]

$$
\min_{0\le t<T}\mathbb{E}\bigl\|\nabla_\phi J(\phi_t)\bigr\|^2\le\frac{16\,L\,\Delta}{T}+4\sqrt{\frac{2L\Delta\sigma_g^2}{T}}+3B^2\rho^{2K}
$$

**含义**: 截断 BPTT 的非渐近收敛保证，右侧三项分别对应优化误差（$T \to \infty$ 时消失）、随机梯度方差项（$T \to \infty$ 时消失）和截断偏差（$K \to \infty$ 时消失）

**符号说明**:
- $L$: 目标函数 Lipschitz 平滑系数
- $\Delta = J(\phi_0) - J^*$: 初始次优性间隙
- $\sigma_g^2$: 梯度估计方差
- $B, \rho$: 截断偏差的界常数（$\rho < 1$ 时指数衰减）
- $K$: 截断深度（$K=2$ 时余弦相似度已超 0.7 阈值，见 Figure 4）

### 公式11: [[Policy-Utility Residual|PUR 诊断]]

$$
\mathrm{PUR} = 1 - \cos\!\left(\nabla_\phi \mathcal{L}_{\mathrm{MSE}}(\phi),\;-\nabla_\phi J(\pi_{\theta^+(\phi)})\right)
$$

**含义**: 度量像素 MSE 梯度方向与元梯度方向的失调程度；PUR = 0 表示完全对齐，PUR = 2 表示完全反向；低 PUR 与高策略效用强相关（Spearman ρ = −0.89）

**符号说明**:
- $\cos(\cdot,\cdot)$: 余弦相似度
- PUR ∈ [0, 2]：越低越好

---

## 关键图表

### Figure 1: DRILL 系统概览

![Figure 1 - DRILL Overview](https://arxiv.org/html/2610.30814v1/drill.png)

**说明**: DRILL 的双路训练机制。策略首先在当前[[世界模型]]内通过虚拟 [[GRPO]] 完成内层更新；该信号反馈至[[世界模型]]参数更新。DRILL-VWI 路径（左）使用虚拟 rollout 的符号加权访问分布构造闭式代理损失；DRILL-IMG 路径（右）通过仿真器收集真实轨迹并[[截断BPTT|截断反向传播]]估计精确元梯度。

### Figure 2: MSE–成功率解耦（LIBERO-Long，π0-3B）

**说明**: 以像素 MSE（×10⁻³）为横轴、任务成功率为纵轴绘制散点图，各方法均标注均值 ± 标准差。关键观察：WMPO（MSE=8.4）比 DRILL-IMG（MSE=9.8）像素更精准，但成功率低 17 个百分点；DRILL-IMG 以**更高** MSE 取得**更高**成功率，直接证明像素目标与策略效用的错配。

### Figure 3: LIBERO-Long 三项诊断

**(a) 学习曲线**：DRILL 变体在早期训练阶段就超越基线并持续改善；WMPO 和 RLVR-World 在更长训练步数后趋于平台。

**(b) Horizon 效应**：在 $H = 20$ 时 DRILL-IMG 仅比 DRILL-VWI 高 1.8 个百分点；随 $H$ 增至 100 该差距扩大至最大值（DRILL-IMG 峰值 83.4），此后虚拟轨迹噪声累积，DRILL-IMG 优势收窄。DRILL-VWI 在 SimplerEnv（短 horizon 任务）略优于 DRILL-IMG（64.0 vs 63.1）。

**(c) PUR 动态**：训练过程中 PUR 的下降轨迹与成功率上升轨迹高度吻合；DRILL-IMG 最终达到 PUR=0.36，而 WMPO 停留在 1.42。

### Figure 4: 截断 HVP 梯度质量（附录）

**说明**: 在二维 gridworld 和 Meta-World MT10 上，随截断深度 $K \in \{1,\ldots,5\}$ 变化，截断 HVP 元梯度与精确元梯度的余弦相似度（±1 std，10 seeds × 1000 states）。$K=2$ 已在两个环境中超过 0.7 阈值；$K \geq 3$ 额外提升不足 0.07 但显著增加计算开销，因此选 $K=2$ 为默认值。

### Figure 5: 虚拟 rollout 质量相图（附录）

**说明**: 以像素 MSE（横轴，×10⁻³，越低越好）和 TRPS（Task-Relevant Prediction Score，纵轴，越高越好）为坐标绘制相图。WMPO 的轨迹偏向低 MSE / 低 TRPS 象限；DRILL-IMG 的轨迹偏向高 MSE / 高 TRPS 象限——再次印证任务相关性与像素精度的错配关系。

### Table 1: 主实验结果（π0-3B，成功率 %，5 seeds 均值±标准差）

| 方法 | LIBERO-Long | CALVIN ABC→D | RoboCasa | MT50 | SimplerEnv | 均值 |
|------|-------------|--------------|----------|------|------------|------|
| Scratch-GRPO | 18.2 | 22.5 | 8.1 | 21.4 | 15.0 | 17.0 |
| Sim-PPO | 28.5 | 32.1 | 12.4 | 31.6 | 22.8 | 25.5 |
| [[WMPO]] | 64.2 | 71.8 | 38.5 | 67.4 | 55.1 | 59.4 |
| RLVR-World | 70.8 | 75.3 | 42.7 | 72.9 | 61.6 | 64.7 |
| **DRILL-VWI** | **76.1** | **80.7** | **51.8** | **75.6** | **64.0** | **69.6** |
| **DRILL-IMG** | **81.3** | **83.1** | **55.7** | **79.4** | **63.1** | **72.5** |
| DRILL-IMG+RLVR-WM | 83.9 | 85.4 | 59.1 | 81.8 | 67.3 | 75.5 |

**关键发现**: DRILL-IMG 比 WMPO 均值高 +13.1 点，比 RLVR-World 高 +7.9 点；与 RLVR-World 预训练叠加后再进一步提升 +3.2 点，说明 DRILL 与现有方法正交互补。像素 MSE：WMPO 8.4、RLVR-World 7.2、DRILL-VWI 9.3、DRILL-IMG 9.8。

### Table 2: VWI 消融（三 benchmark 均值，成功率 %）

| 配置 | 成功率 |
|------|--------|
| WMPO baseline | 58.2 |
| + 访问密度（仅 visitation）| 64.2 |
| + 符号优势幂次（仅 signed advantage）| 66.8 |
| + 世界模型置信度（仅 confidence）| 60.1 |
| **DRILL-VWI（全三项）** | **69.2** |
| α=2（幂次变体）| 67.9 |

**关键发现**: 三项权重组合缺一不可，符号优势幂次单项贡献最大（66.8），完整组合达到最优（69.2）；$\alpha=1$ 优于 $\alpha=2$。

### Table 3: OpenVLA-7B 复现结果（均值成功率 %）

| 方法 | 均值成功率 |
|------|------------|
| WMPO | 53.8 |
| RLVR-World | 60.4 |
| **DRILL-IMG** | **68.9** |

**关键发现**: DRILL-IMG 在 OpenVLA-7B 骨干上复现了 +15.1 点相对 WMPO 的提升，说明方法不依赖特定骨干。

### Table 4: 世界模型规模扩展效应

| 世界模型规模 | WMPO | DRILL-IMG | 差值 |
|-------------|------|-----------|------|
| 300M | 55.2 | 63.1 | +7.9 |
| 1.2B | 59.4 | 72.5 | +13.1 |
| 7B | 62.8 | 79.2 | +16.4 |

**关键发现**: DRILL-IMG 相对 WMPO 的增益随世界模型规模增大而扩大，表明更强大的世界模型更能从目标导向的训练目标中获益。

---

## 实验

### 数据集

| 数据集 / 仿真器 | 规模 | 特点 | 用途 |
|----------------|------|------|------|
| LIBERO-Long | 50 任务 | 长视程桌面操作，需连续多步骤 | 主要 benchmark |
| CALVIN ABC→D | 34 任务 | 跨场景泛化，ABC 训练→D 测试 | 泛化评估 |
| RoboCasa | 100 任务 | 厨房场景多样操作 | 复杂操作 |
| Meta-World MT50 | 50 任务 | 多任务操作基准 | 多任务能力 |
| SimplerEnv | 25 任务 | Google Robot 真实机器人仿真 | 真实感评估 |

### 实现细节

- **VLA 骨干**: π0-3B（主实验）/ OpenVLA-7B（复现）
- **世界模型**: 300M / 1.2B / 7B 三种规模，Gaussian observation head
- **DRILL-VWI**: 无额外仿真器调用；kNN buffer 大小 50k
- **DRILL-IMG 截断深度**: $K=2$；真实 rollout 批量 $B_{\mathrm{real}}=30$
- **内层学习率**: $\eta_\theta = 2\times10^{-5}$
- **优化器**: AdamW，外层 $\eta_\phi = 1\times10^{-4}$
- **Rollout Horizon**: $H=100$（主实验最佳）
- **硬件**: 24 × H20 GPU，约 4 周（~17k H20-hours）

### 可视化结果

LIBERO-Long 上 DRILL-IMG 的 TRPS（Task-Relevant Prediction Score）达到 0.81，而 WMPO 仅 0.61，说明目标导向训练后[[世界模型]]对任务关键区域的预测精度显著提升。120 次双盲人类评估中，评估者在展示任务完成情况后有 62% 偏好 DRILL-IMG 的轨迹（$p < 0.001$）。在 LIBERO 扰动测试中，DRILL-IMG 的平均相对成功率下降为 18.6%，而 WMPO 为 30.2%，显示出更好的鲁棒性。

---

## 批判性思考

### 优点

1. **目标对齐的理论动机清晰**: 像素 MSE 与策略梯度信号的错配问题在实验中有直接证据（17% 更高 MSE 带来更好策略）
2. **双变体设计权衡合理**: VWI 无需仿真器交互、计算轻量；IMG 精度更高但成本更大，两者适用不同资源约束场景
3. **收敛保证严格**: Theorem 1 给出非渐近界，同时分析了优化误差、方差和截断偏差三类来源
4. **PUR 通用性**: PUR 作为诊断工具可推广到其他世界模型训练方法的评估，Spearman ρ=−0.89 相关性强

### 局限性

1. **仅有仿真实验**: 全部结果在 5 个仿真器上获得，缺乏真实机器人实验；仿真器的奖励函数与感知保真度未必迁移到现实
2. **DRILL-IMG 计算成本高**: 截断 BPTT + 仿真器 rollout 需要 ~17k H20-hours，部署门槛高
3. **理论保证局部且局部子问题化**: Theorem 1 是对每个子问题的局部保证，不覆盖联合 $(\theta, \phi)$ 进程的全局收敛
4. **奖励模型依赖**: 需要一个有效的冻结 VLM 奖励模型，奖励稀疏或不准确时效果未知

### 潜在改进方向

1. 真实机器人验证，特别是 RoboCasa / SimplerEnv 中的任务
2. 将截断深度 $K$ 自适应调整（当前固定为 2），减少不必要的 HVP 计算
3. 将 PUR 作为在线监控信号，动态切换 VWI 和 IMG 策略以平衡效率与精度

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（论文附录包含完整超参数，含 B_real、K、η 等）
- [x] 数据集 / 仿真器可获取（LIBERO、CALVIN、MT50、SimplerEnv 均开源）

---

## 关联笔记

### 基于

- [[WMPO]]: DRILL 的直接基线，DRILL 在其双阶段框架上替换世界模型目标
- [[GRPO]]: 内层策略优化算法，DRILL 依赖 GRPO 构造内层更新
- [[世界模型]]: DRILL 所属的大类方法——为 VLA 提供虚拟 rollout 的视频预测模型
- [[MAML]]: DRILL-IMG 的设计灵感来源，元梯度通过内层更新的反向传播估计

### 对比

- [[VLAW]]: 同为 WMPO 改进工作，VLAW 聚焦额外正则化而非目标替换
- [[DreamerV3]]: 早期价值感知世界模型学习方法（VAML 系列），DRILL 在 VLA 场景下实现类似对齐
- [[MuZero]]: 通过可规划表示隐式实现目标对齐，但不直接优化策略改进方向

### 方法相关

- [[双层优化]]: DRILL 的核心优化框架，外层优化世界模型、内层优化策略
- [[元梯度]]: DRILL-IMG 的梯度估计机制，通过截断 BPTT 实现
- [[截断BPTT]]: DRILL-IMG 限制反向传播步数（K=2）的具体技术
- [[Policy-Utility Residual]]: DRILL 提出的诊断工具，度量世界模型目标与策略效用的对齐度
- [[GRPO]]: 内层策略更新算法

### 硬件 / 数据相关

- [[π0]]: 主实验使用的 3B VLA 骨干
- [[OpenVLA]]: 复现实验使用的 7B VLA 骨干
- [[LIBERO]]: 主要 benchmark，长视程操作
- [[CALVIN]]: 跨场景泛化 benchmark
- [[Meta-World]]: 多任务操作 benchmark

---

## 速查卡片

> [!summary] DRILL: Training World Models to Improve Policies, Not Predict Pixels
> - **核心**: [[世界模型]]像素 MSE 目标与策略梯度信号存在根本错配，DRILL 用双层目标替换之
> - **方法**: DRILL-VWI（三项加权代理，无需额外仿真器）+ DRILL-IMG（K=2 截断 BPTT 元梯度，30 real rollouts/step）
> - **结果**: π0-3B 在 5 仿真器均值成功率 +13.1 vs WMPO，+7.9 vs RLVR-World；PUR 从 1.42→0.36
> - **代码**: 未公开（NeurIPS 2026 版本）

---

*笔记创建时间: 2026-10-24*
