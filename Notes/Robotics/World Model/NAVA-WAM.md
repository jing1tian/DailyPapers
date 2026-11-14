---
title: "Native Action-Prior Learning from Videos for World Action Models"
method_name: "NAVA-WAM"
authors: [Zhaochong An, Fei Zhang, Menglin Jia, Duncan Frost, Zijian Zhou, Yikai Wang, Xudong Wang, Aditya Patel, Belinda Zeng, Tao Xiang, Serge Belongie, Amir Bar, Sen He]
year: 2026
venue: arXiv
tags: [world-action-model, video-pretraining, flow-matching, action-prior-learning, mixture-of-transformers, robot-manipulation, vla]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.03391
created: 2026-11-14
---

# 论文笔记：Native Action-Prior Learning from Videos for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Meta AI; University of Copenhagen; University of Surrey |
| 日期 | October 2026 |
| 项目主页 | [https://zhaochongan.github.io/projects/NAVA-WAM](https://zhaochongan.github.io/projects/NAVA-WAM) |
| 对比基线 | [[π0]], [[π0.5]], [[Fast-WAM]], [[Image-WAM]], [[MV-WAM]], [[HALO]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.03391) / Code 未开源 |

---

## 一句话总结

> NAVA-WAM 通过将 [[Action-DiT]] 直接在无标注观测视频上用未来帧 [[Flow Matching|流匹配]] 监督进行预训练，使动作策略直接习得转移先验，无需中间的视觉表示适配或隐动作模型，在仿真与真实机器人任务上全面超越基线。

---

## 核心贡献

1. **原生动作先验预训练（Native Action-Prior Learning）**: 直接在无动作标签的观测视频上训练 [[Action-DiT]]，将未来视频 [[Flow Matching|流匹配]] 损失反传到动作路径，无需构造隐动作变量
2. **转移结构化注意力（Transition-Structured Attention）**: 预训练阶段动作 token 同时看到干净的前帧与加噪的后帧，使动作流习得帧间转移信息；未来帧也反向读取动作表示
3. **非对称注意力推理（Asymmetric Attention Inference）**: 后训练阶段 [[Video-DiT]] 仅自注意，[[Action-DiT]] 单向读取视觉 token；推理时一次性缓存视觉键值对，仅对动作去噪，显著降低推理开销

---

## 问题背景

### 要解决的问题

[[World Action Model|世界动作模型（WAM）]] 将未来视觉预测与机器人动作预测联合建模，但训练需要大量带动作标签的机器人轨迹数据，标注成本高昂。另一方面，互联网上存在海量无标签的观测视频（如第一人称视频、机器人 demo 录像），如何高效利用这些数据提升策略性能是关键问题。

### 现有方法的局限

现有方法利用无标签视频的路径有两条（见 Figure 1）：

- **视觉表示学习路径**（如 [[π0.5]]）：先用视频预训练视觉骨干，后期将表示迁移到控制任务。表示空间为视觉设计，未必对控制有用；迁移时存在域适应问题。
- **隐动作模型路径**（如 [[DreamZero]]）：先从视频中推断隐动作变量，再学习从隐动作到真实动作的映射，引入额外建模阶段和误差传播。

两种路径都存在"视频预训练→控制后训练"的断层：视频目标与动作目标不一致。

### 本文的动机

NAVA-WAM 提出第三条路径：**原生动作先验学习**——将动作策略网络（[[Action-DiT]]）直接插入视频预训练过程，用未来帧的流匹配损失直接塑造动作路径的参数，从而消除中间步骤，使预训练目标与下游控制目标内在一致。

---

## 方法详解

### 模型架构

NAVA-WAM 采用 **[[Mixture-of-Transformers]]（MoT）** 架构，包含两个专家：

- **输入**: 语言指令 $l$ + 当前观测 $o_t$（通过冻结 [[VAE]] 编码为潜变量 $z$）+ 状态/本体感受 $q_t$
- **[[Video-DiT]]**: 预训练好的视频生成 Transformer，处理视觉流；预训练阶段**冻结**
- **[[Action-DiT]]**: 1B 参数可训练 Transformer，处理动作流；与 Video-DiT 通过**联合注意力**交换信息
- **输出**: [[Action Chunking|动作块]] $a_{t:t+H-1}$（horizon $H=16$）
- **总参数**: ~1B 可训练（Action-DiT），Video-DiT 参数量更大但预训练阶段冻结

两个专家共享 Transformer 层深但参数独立，通过 **joint attention** 在每一层互相读取对方的键值对。

### 核心模块

#### 模块1：转移结构化联合注意力（Transition-Structured Attention）

**设计动机**: 要让 Action-DiT 从视频中学到转移先验，需要同时看到 $z_{k-1}$（前帧干净状态）和 $z_{k,\tau_v}$（后帧加噪状态），才能在去噪过程中推断帧间的运动转移信息。加噪设计防止 Action-DiT 直接复制目标帧（Proposition A.2 理论证明）。

**具体实现**（预训练阶段）：

- **Action-DiT** 的注意力：
$$H_{A,k} = \operatorname{Attn}\left(Q_{A,k},\; [K_{V,k-1};\, K_{V,k,\tau_v};\, K_{A,k}],\; [V_{V,k-1};\, V_{V,k,\tau_v};\, V_{A,k}]\right)$$
动作 token 的查询同时关注干净前帧、加噪后帧和自身动作 token 的键值对。

- **Video-DiT**（加噪后帧）的注意力：
$$H_{V,k,\tau_v} = \operatorname{Attn}\left(Q_{V,k,\tau_v},\; [K_{V,k-1};\, K_{V,k,\tau_v};\, K_{A,k}],\; [V_{V,k-1};\, V_{V,k,\tau_v};\, V_{A,k}]\right)$$
加噪后帧也能读取动作 token，使视频损失梯度可以流过动作路径。

#### 模块2：非对称注意力（Asymmetric Attention，后训练阶段）

**设计动机**: 推理时不需要生成未来视频，只需要动作去噪。若 Video-DiT 在后训练中也使用 joint attention（双向），推理时必须同时运行 Video-DiT，代价高。非对称设计令 Video-DiT 仅自注意，使其键值对可被缓存，动作去噪时只需运行 Action-DiT。

**具体实现**（后训练阶段）：

- **Video-DiT** 的注意力（仅自注意）：
$$H_V = \operatorname{Attn}(Q_V,\; K_V,\; V_V)$$

- **Action-DiT** 的注意力（读取 Video-DiT 输出，单向）：
$$H_A = \operatorname{Attn}(Q_A,\; [K_V;\, K_A],\; [V_V;\, V_A])$$

此设计允许推理时一次计算并缓存视觉键值对 $\mathcal{C}_V = \{K_\ell^V,\, V_\ell^V\}_{\ell=1}^L$，然后仅执行动作去噪 ODE。

---

## 关键公式

### 公式1：[[Action Chunking|策略定义]]

$$
\mathcal{P}_{\theta}(a_{t:t+H-1} \mid o_t,\, l,\, q_t)
$$

**含义**: 给定当前观测 $o_t$、语言指令 $l$、本体感受状态 $q_t$，预测未来 $H$ 步的动作块。

**符号说明**:
- $a_{t:t+H-1}$: 动作块，horizon $H=16$
- $o_t$: 当前时刻观测（由 VAE 编码为潜变量 $z$）
- $l$: 自然语言任务指令
- $q_t$: 本体感受状态（关节角度等）

---

### 公式2：[[Joint Attention|联合注意力机制]]

$$
H_m = \operatorname{Attn}\left(Q_m,\; [K_V;\, K_A],\; [V_V;\, V_A]\right), \quad m \in \{V, A\}
$$

**含义**: 视觉专家和动作专家的每个 head 均跨流关注对方的键值对，实现信息双向交换。

**符号说明**:
- $Q_m$: 专家 $m$ 的查询矩阵
- $[K_V; K_A]$: 拼接的视觉和动作键矩阵
- $[V_V; V_A]$: 拼接的视觉和动作值矩阵

---

### 公式3：[[Flow Matching|预训练流匹配噪声构造]]

$$
z_{k,\tau_v} = (1 - \tau_v)\,z_k + \tau_v\,\epsilon_k^v, \qquad u_k^v = \epsilon_k^v - z_k
$$

**含义**: 对第 $k$ 个未来帧潜变量按时间步 $\tau_v$ 进行线性插值加噪，目标速度场 $u_k^v$ 为噪声减去原始数据。

**符号说明**:
- $z_k$: 第 $k$ 段的干净未来帧潜变量
- $\tau_v \sim \mathcal{U}(0,1)$: 视频流的噪声时间步（越大越嘈杂）
- $\epsilon_k^v \sim \mathcal{N}(0, I)$: 随机高斯噪声
- $u_k^v$: 视频流的目标速度场

---

### 公式4：[[Flow Matching|预训练损失]]

$$
\mathcal{L}_{\mathrm{pre}} = \mathbb{E}\!\left[\sum_{k=1}^{K} w(\tau)\left\|v^v_{\bar{\theta}_V,\theta_A}\!\left([z_{k-1}, z_{k,\tau_v}], \epsilon_k^a, \bm{\tau} \mid l\right) - u_k^v\right\|_2^2\right]
$$

**含义**: 仅用视频流的流匹配损失训练 Action-DiT 参数 $\theta_A$，Video-DiT 参数 $\bar{\theta}_V$ 冻结（$\bar{}$ 表示 stop-gradient）。

**符号说明**:
- $v^v_{\bar{\theta}_V, \theta_A}$: 对视频流建模的速度场，Video-DiT 冻结
- $w(\tau)$: 时间步加权系数
- $K$: 视频转移段数
- $\epsilon_k^a$: 动作流随机噪声（作为 Action-DiT 的初始输入）
- $\bm{\tau}$: 时间步向量 $[\tau_v, \tau_a]$

---

### 公式5：[[Flow Matching|后训练联合损失]]

**后训练流匹配构造**（视频和动作双流）:

$$
z_{k,\tau_v} = (1-\tau_v)z_k + \tau_v\epsilon_k^v,\; u_k^v = \epsilon_k^v - z_k
$$

$$
a_{k,\tau_a} = (1-\tau_a)a_k + \tau_a\epsilon_k^a,\; u_k^a = \epsilon_k^a - a_k
$$

**后训练损失**:

$$
\mathcal{L}_{\mathrm{FM}}^{m} = \mathbb{E}\!\left[\sum_{k=1}^{K} w_m(\tau_m)\left\|v^m_{\theta_V,\theta_A}\!\left([z_{k-1}, z_{k,\tau_v}], a_{k,\tau_a}, \bm{\tau} \mid l, q\right) - u_k^m\right\|_2^2\right],\; m \in \{v, a\}
$$

$$
\mathcal{L}_{\mathrm{post}} = \lambda_v \mathcal{L}_{\mathrm{FM}}^v + \lambda_a \mathcal{L}_{\mathrm{FM}}^a
$$

**含义**: 视频流和动作流联合以流匹配损失训练，两个专家均可更新，$\lambda_v, \lambda_a$ 为权重系数。

**符号说明**:
- $a_{k,\tau_a}$: 加噪动作
- $\tau_a$: 动作流的噪声时间步
- $\lambda_v, \lambda_a$: 视频流和动作流损失权重

---

### 公式6：[[KV-Cache|视觉缓存]]与[[推理]]

**视觉键值缓存**:

$$
\mathcal{C}_V = \{K_\ell^V,\, V_\ell^V\}_{\ell=1}^{L}
$$

**动作-only 去噪 ODE**（$S=10$ Euler 步）:

$$
\frac{da_{\tau_a}}{d\tau_a} = v^a_{\theta_A}(a_{\tau_a},\, \tau_a \mid \mathcal{C}_V,\, l,\, q),\quad \tau_a: 1 \to 0
$$

**含义**: 推理时仅运行 Action-DiT，以缓存的视觉键值对 $\mathcal{C}_V$ 为条件执行动作去噪，整个视频生成过程被跳过。

---

## 关键图表

### Figure 1：三种利用观测视频的策略学习路径对比

![Figure 1](https://arxiv.org/html/2610.03391/2610.03391v2/fig1.png)

**说明**: 从左到右：(a) **[[视觉表示学习]]路径**——用视频预训练视觉骨干，冻结后接控制头，视觉与控制目标不对齐；(b) **[[隐动作模型]]路径**——额外学习一个 latent-action 模型推断隐动作，引入二阶段误差；(c) **原生动作先验学习路径（本文）**——Action-DiT 直接在视频预训练中更新，视频损失即动作损失，无中间步骤。

---

### Figure 2：NAVA-WAM 整体架构与三阶段流程

![Figure 2](https://arxiv.org/html/2610.03391/2610.03391v2/main.png)

**说明**: 展示 [[Mixture-of-Transformers]] 双专家结构以及预训练/后训练/推理三阶段。预训练阶段：[[Video-DiT]] 冻结（灰色），只更新 [[Action-DiT]]（蓝色），动作 token 通过 [[Transition-Structured Attention|转移结构化注意力]] 从加噪未来帧中学习转移先验。后训练阶段：双流均更新，[[Asymmetric Attention|非对称注意力]] 令 Video-DiT 仅自注意。推理阶段：缓存视觉 KV，只对动作去噪，无视频生成开销。

---

### Figure 3：真实机器人 Franka FR3 评估

![Figure 3](https://arxiv.org/html/2610.03391/2610.03391v2/real_robot.png)

**说明**: 三个任务上 NAVA-WAM（93.3%）vs. [[DreamZero]]（66.7%）vs. [[π0.5]]（53.3%）的成功率对比与 rollout 示例。NAVA-WAM 在 T1（将方块移到碗左侧）上表现尤为突出（4/5），而其他两个基线均 0/5。推理延迟 379.2 ms/call，远低于 DreamZero 的 3523.9 ms，接近 π0.5 的 195.7 ms。

---

### Figure 4：学到的动作表示对视觉 token 的注意力分布

![Figure 4](https://arxiv.org/html/2610.03391/2610.03391v2/figure/fig_tab2b_attention.png)

**说明**: 对比 NAVA-WAM（原生先验预训练）与隐动作基线的注意力热图。NAVA-WAM 的动作 token 更聚焦于任务相关的物体区域（机械臂末端、目标物），而非背景，说明学到了更有语义的转移先验。

---

### Figure 5：跨域动作转移（Motion Transfer）可视化

![Figure 5](https://arxiv.org/html/2610.03391/2610.03391v2/figure/fig_motion_transfer.png)

**说明**: 将源视频转移段的动作表示 $r_k = q(z_{k-1}, z_k)$ 应用于不同目标初始帧 $z'_{k-1}$，生成 $\hat{z}'_k = f(z'_{k-1}, r_k)$。跨仿真/真实/sim-to-real 场景，动作表示成功迁移物体运动轨迹，验证其捕获了域不变的转移信息。

---

### Figure 6：观测-only 预训练数据规模消融

（来源于论文正文，URL 不可用，详见 arXiv HTML）

**说明**: 在不同动作标签 budget 下（20/50/100/full demonstrations），随观测-only 预训练数据规模增大（0.1×→0.5×→1.0×），下游策略性能持续提升，表明 NAVA-WAM 能有效利用更多无标签视频。

---

### Figure 7：转移结构化注意力掩码（预训练与后训练对比）

（来源于附录 B，URL 不可用，详见 arXiv HTML Fig. 7）

**说明**: 可视化三段连续帧（$k-1, k, k+1$）的注意力掩码设计：预训练阶段动作 token 跨段关注前后帧视觉 token；后训练阶段（非对称）Video-DiT 只自注意，Action-DiT 单向读取视觉 token。

---

### Figure 8：原生动作先验预训练动态曲线

（来源于附录 C，URL 不可用，详见 arXiv HTML Fig. 8）

**说明**: 展示预训练阶段训练集和 held-out 未来视频流匹配损失随步数下降曲线，验证 Action-DiT 确实从无标签视频中持续学习，无过拟合。

---

### Figure 9：held-out 视频未来帧预测质量

![Figure 9](https://arxiv.org/html/2610.03391/2610.03391v2/pretrain_recon.png)

**说明**: 四组 held-out 第一人称视频（真值 vs. NAVA-WAM 预测未来帧）的定性对比，确认习得的动作先验具备合理的视频生成能力。

---

### Figure 10：更多跨域动作转移示例

![Figure 10](https://arxiv.org/html/2610.03391/2610.03391v2/figure/fig_motion_transfer_additional.png)

**说明**: 仿真→真实、真实→仿真、sim-to-real 多个域的更多 motion transfer 案例，补充验证动作表示的域无关性。

---

### Figure 11：RoboTwin 2.0 随机域 rollout 可视化（成功与失败案例）

![Figure 11](https://arxiv.org/html/2610.03391/2610.03391v2/robotwin_qualitative.png)

**说明**: 7 个成功案例（包括 Handover Mic、Lift Pot 等）和 2 个失败案例（Move Can Pot、Hanging Mug——需要精细接触的操作），直观展示 NAVA-WAM 在随机扰动域的鲁棒性及局限。

---

### Table 1：LIBERO 各子集成功率（%）

| Method | Spatial | Object | Goal | Long | **Avg.** |
|--------|---------|--------|------|------|----------|
| [[π0]] | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 |
| [[π0.5]] | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| [[Fast-WAM]] | 98.2 | 100.0 | 97.0 | 95.2 | 97.6 |
| [[Image-WAM]] | 97.2 | 99.2 | 98.8 | 98.4 | 98.4 |
| **NAVA-WAM** | **98.8** | **99.6** | **98.6** | **99.0** | **99.0** |

**关键发现**: NAVA-WAM 在 Long 子集上提升最显著（99.0% vs. Image-WAM 98.4%），说明世界模型预测能力对长时序任务尤为重要。

---

### Table 2：LIBERO-Plus 各扰动维度成功率（%）

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg.** |
|--------|--------|-------|----------|-------|------------|-------|--------|----------|
| [[π0]] | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 |
| [[π0-FAST]] | 65.1 | 21.6 | 61.0 | 73.2 | 73.2 | 74.4 | 68.8 | 61.6 |
| [[Fast-WAM]] | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 |
| [[Image-WAM]] | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 |
| **NAVA-WAM** | **72.8** | **65.6** | **93.8** | **97.9** | **82.8** | **84.8** | **87.1** | **83.5** |

**关键发现**: NAVA-WAM 在 Robot（机器人形态变化）和 Layout（场景布局变化）两个最难扰动维度上显著领先所有基线，说明 video pretraining 赋予了对环境变化更强的泛化能力。

---

### Table 3：RoboTwin 2.0 整体成功率（%，50 任务平均）

| Method | Clean avg. | Random avg. |
|--------|-----------|-------------|
| DP | 28.0 | 0.6 |
| RDT | 34.5 | 13.7 |
| [[π0]] | 46.4 | 16.3 |
| UP-VLA | 52.9 | 15.2 |
| [[BagelVLA]] | 75.3 | 20.5 |
| [[HALO]] | 80.5 | 26.4 |
| [[Fast-WAM]] | 71.9 | 6.3 |
| [[MV-WAM]] | 84.0 | 55.7 |
| **NAVA-WAM** | **88.5** | **73.6** |

**关键发现**: Random 域（随机物理扰动）上 NAVA-WAM（73.6%）比第二名 MV-WAM（55.7%）高出 **17.9 个百分点**，Clean 域领先幅度 4.5 pp。Random/Clean 比值（83.1%）远高于其他方法（MV-WAM 66.3%），说明 video pretraining 在分布外泛化上价值显著。

---

### Table 4：真实机器人（Franka FR3）成功率

| Model | ms/call | T1 | T2 | T3 | **Avg. (%)** |
|-------|---------|----|----|-----|-------------|
| [[π0.5]] | 195.7 | 0/5 | 5/5 | 3/5 | 53.3 |
| [[DreamZero]] | 3523.9 | 0/5 | 5/5 | 5/5 | 66.7 |
| **NAVA-WAM** | 379.2 | **4/5** | **5/5** | **5/5** | **93.3** |

T1 = 将方块移到碗左侧；T2 = 将香蕉放进盒子；T3 = 将蓝色方块叠放在红色方块上

**关键发现**: NAVA-WAM 在最难任务 T1 上（需要精确定位）以 4/5 大幅领先其他两个基线（均为 0/5），且推理延迟只有 DreamZero 的 1/9，接近 π0.5 的 2 倍（可接受范围）。

---

### Table 5：消融实验——预训练策略对比

（详见论文附录 C，图 6 数据：不同动作标签 budget 下的性能）

| 预训练策略 | Full demo budget 性能 | 低 budget（20 demos）性能 |
|-----------|----------------------|--------------------------|
| 无预训练（scratch）| 低 | 更低 |
| 视觉表示学习 | 中 | 中 |
| 隐动作模型 | 较高 | 较高 |
| **NAVA（本文）** | **最高** | **最高** |

**关键发现**: 在所有动作标签 budget 下，原生动作先验学习均优于视觉表示学习和隐动作模型路径，且低标注场景下优势更明显——验证了方法在数据高效性上的价值。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[DROID]] / 第一人称视频 | ~235,790 episodes（1,054.3h）| 无动作标签 | 预训练 |
| [[LIBERO]] | 4 子集（Spatial/Object/Goal/Long） | 50 demos/任务 | 后训练 + 评测 |
| LIBERO-Plus | 7 扰动维度 | 分布外泛化测试 | 评测 |
| [[RoboTwin 2.0]] | 50 任务，Clean + Random 域 | 随机物理扰动 | 评测 |
| Franka FR3 真实机器人 | 3 任务，15 trials | 仿真到现实迁移 | 评测 |

### 实现细节

- **预训练**: AdamW，学习率 $1 \times 10^{-4}$，global batch size 512，14,000 steps，128 GPU
- **后训练（LIBERO）**: Video-DiT LR $4 \times 10^{-5}$（冷启动）/ $1 \times 10^{-4}$（联合），Action-DiT LR $2 \times 10^{-4}$，cosine scheduler，5% warm-up
- **后训练（RoboTwin）**: Video-DiT LR $2 \times 10^{-4}$（冷启动）/ $1 \times 10^{-4}$（联合），Action-DiT LR $3 \times 10^{-4}$
- **Action horizon**: $H = 16$
- **推理去噪步数**: $S = 10$（Euler ODE solver）
- **硬件**: 128 × GPU（型号未明确）

### 可视化结果

Motion transfer 实验（Figure 5、10）定性证明 Action-DiT 学到的 $r_k$ 编码了域无关的运动转移信息，可跨仿真/真实域迁移物体轨迹。RoboTwin rollout 失败案例集中于需要精细接触的任务（如挂钩），与模型当前局限一致。

---

## 批判性思考

### 优点

1. **原理简洁，无额外建模阶段**: 相比隐动作模型路径，无需设计和训练独立的隐动作模块，端到端更优雅
2. **随机域泛化能力突出**: RoboTwin 2.0 Random 域领先 17.9 pp，表明 video pretraining 能提升策略对场景扰动的鲁棒性
3. **推理高效**: [[Asymmetric Attention|非对称注意力]] + [[KV-Cache|视觉缓存]] 使推理时无需生成视频，4.30 TFLOPs（与其他高效 WAM 相当），实际控制延迟 379.2 ms/call

### 局限性

1. **细粒度接触操作困难**: 在 Hanging Mug、Move Can Pot 等需要精确接触的任务上失败，模型缺少显式的接触力反馈建模
2. **局部执行错误恢复机制缺失**: 模型无法主动从局部执行失败中恢复，依赖初始状态对齐
3. **Video-DiT 仍需后训练更新**: 预训练阶段 Video-DiT 冻结是合理的，但后训练仍需更新双流，计算代价高于只训练 Action-DiT

### 潜在改进方向

1. 引入触觉/力觉传感器信号作为额外模态，改善精细接触操作
2. 设计显式的错误恢复机制（如在线 replanning 或 recovery policy）
3. 探索将 Video-DiT 在后训练阶段也部分冻结，进一步降低计算开销

### 可复现性评估

- [ ] 代码开源（暂未开源）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（附录 B 详细给出超参数）
- [x] 数据集可获取（LIBERO、RoboTwin 2.0 均公开）

---

## 关联笔记

### 基于

- [[Video-DiT]]: 预训练好的视频生成 Transformer，作为视觉专家骨干
- [[Flow Matching]]: 双流训练的核心优化框架
- [[Mixture-of-Transformers]]: 多专家 Transformer 架构基础

### 对比

- [[π0.5]]: 视觉表示学习路径代表，真实机器人成功率 53.3% vs. NAVA-WAM 93.3%
- [[DreamZero]]: 隐动作模型路径代表，推理延迟 3523.9 ms vs. NAVA-WAM 379.2 ms
- [[Fast-WAM]]: 高效 WAM baseline，RoboTwin Random 仅 6.3% vs. 73.6%
- [[MV-WAM]]: 此前 RoboTwin SOTA，Random 55.7% vs. 73.6%
- [[HALO]]: RoboTwin 2.0 Clean 80.5% vs. 88.5%

### 方法相关

- [[Action Chunking]]: 动作块预测策略，horizon H=16
- [[KV-Cache]]: 视觉缓存机制使推理去除视频生成
- [[Asymmetric Attention]]: 非对称注意力设计，后训练阶段 Video-DiT 单向
- [[Transition-Structured Attention]]: 转移结构化注意力，预训练阶段核心设计

### 硬件/数据相关

- [[Franka FR3]]: 真实机器人实验平台
- [[LIBERO]]: 仿真操控 benchmark
- [[RoboTwin 2.0]]: 大规模随机扰动仿真 benchmark
- [[DROID]]: 大规模第一人称机器人视频数据集（用于无标注预训练）

---

## 速查卡片

> [!summary] NAVA-WAM (arXiv 2610.03391)
> - **核心**: 直接在无标注观测视频上用流匹配损失预训练 Action-DiT，无需隐动作变量
> - **方法**: Mixture-of-Transformers（Video-DiT + Action-DiT）+ 转移结构化注意力 + 非对称注意力推理
> - **结果**: LIBERO 99.0% / LIBERO-Plus 83.5% / RoboTwin 2.0 Random 73.6%（+17.9 pp）/ 真实机器人 93.3%
> - **代码**: 暂未开源，项目主页 [https://zhaochongan.github.io/projects/NAVA-WAM](https://zhaochongan.github.io/projects/NAVA-WAM)

---

*笔记创建时间: 2026-11-14*
