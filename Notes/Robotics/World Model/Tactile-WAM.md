---
title: "Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention"
method_name: "Tactile-WAM"
authors: [Siyu Wu, Linjing You, Junjie Zhu, Yaozu Liu, Changhao Zhang, Jian Liu, Weiqiang Wang, Qi Li, Jituo Li, Hengshuang Zhao]
year: 2026
venue: arXiv (RSS 2026 Workshop on Tactile and Foundation Models)
tags: [world-action-model, tactile-sensing, attention-mechanism, contact-rich-manipulation, diffusion-transformer, visuotactile]
zotero_collection: Robotics/World Model
image_source: mixed
arxiv_html: https://arxiv.org/html/2606.26663
created: 2026-06-27
---

# 论文笔记：Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Ant Group；浙江大学；香港大学；深圳湾实验室（Shenzhen Loop Area Institute）；中国科学院自动化研究所 |
| 日期 | June 2026 |
| 项目主页 | 无（RSS 2026 Workshop 投稿，未发现独立项目页/代码仓库） |
| 对比基线 | [[DreamZero]]、[[pi0.5|π₀.₅]]、[[Dream-Tac]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.26663) / [HTML](https://arxiv.org/html/2606.26663) |

---

## 一句话总结

> 提出 TAAM 非对称注意力机制，用 VideoClean 掩码隔离触觉对视觉预测的污染、用触觉变化偏置引导动作去噪，让 [[World Action Model|WAM]] 在接触密集任务上成功率提升 38.9%（接触为主任务提升 86%）。

---

## 核心贡献

1. **触觉感知 WAM 建模**: 将接触密集型操作建模为未来视觉隐变量、未来触觉接触状态、动作块的联合预测问题
2. **触觉污染（Tactile Pollution）现象识别与 TAAM**: 发现无约束的触觉 token 注入会损害视频预测质量，提出 [[Tactile Asymmetric Attention Mechanism (TAAM)|TAAM]]，组合 [[VideoClean Attention Mask|VideoClean]] 掩码与触觉感知偏置
3. **触觉感知动作去噪**: 用预测的触觉变化代理（touch-change proxy）调制动作对触觉 token 的注意力，仅作用于 action-query 到 tactile-key 的路径
4. **仿真 + 真机双重验证**: 在 ManiFeel 仿真基准和真机接触密集任务上系统评估，并通过消融实验验证各组件贡献

---

## 问题背景

### 要解决的问题

[[World Action Model|World Action Models (WAMs)]] 将动作生成与未来状态预测耦合，让策略推理"一个动作块会引发什么"，而不只是"该输出什么动作"。但插入、装配、搜索、重定向等接触密集型任务依赖接触开始、滑动、卡阻、接触法线、毫米级对齐误差等物理量，这些变量在 RGB 图像中往往弱可见、被遮挡，甚至完全不可见——因此一个视觉上看似合理的未来 rollout，物理上可能是不完整的。

### 现有方法的局限

- **纯视觉 WAM**（如 [[DreamZero]]）：预测的"世界"仅由未来场景外观表示，无法捕捉接触状态
- **现有触觉 WAM/视触觉世界模型**（[[Visuo-Tactile World Models|VT-WM]]、[[OmniVTA]]、[[VTAM]]、[[DreamTacVLA]]、[[TacForeSight]]、[[Dream-Tac]]）：已证明触觉是有价值的预测信号，但对于"视觉预训练的 WAM"，留下两个注意力层面的问题未解决：
  1. **触觉污染（Tactile Pollution）**: 触觉信号稀疏、局部、事件驱动，若允许触觉 token 无约束地参与所有注意力路径，会迫使视觉动力学模型吸收对未来视频弱预测性的信号，从而降低视频预测质量或导致视觉主导行为退化
  2. **触觉感知不完整**: 触觉 WAM 应该知道"触觉何时与动作相关"——最有用的触觉信号通常是阶跃式的接触变化（首次接触、滑动开始、压缩变化、卡阻），而不仅仅是触觉 token 的存在本身

### 本文的动机

作者认为关键不是"触觉是否有用"（这点已被前人工作证明），而是"触觉该被放在计算图的哪个位置"：应该让触觉自由进入所有 token 交互、停留为策略侧修正信号，还是成为一个仅在接触相关时才被路由到动作生成路径的**预测物理状态**？Tactile-WAM 选择第三种路径——触觉作为受接触动力学监督的生成式物理状态，通过非对称动作路径路由，而非仅作为 grounding 特征或无限制的跨模态注意力源。

---

## 方法详解

### 模型架构

Tactile-WAM 采用 **单一 Wan/WAM 兼容去噪骨干网络（Diffusion Transformer）** 在联合 token 序列上去噪：
- **输入**: 语言指令 $l$ + 视觉历史 $o^v_{\le t}$ + 触觉历史 $o^\tau_{\le t}$ + 本体状态 $s_{\le t}$
- **Backbone**: [[Wan]] 视频生成骨干改造的 [[Diffusion Transformer|DiT]]（"Causal Wan DiT"）
- **核心模块**: [[Tactile Asymmetric Attention Mechanism (TAAM)|TAAM]] 用于 [[VideoClean Attention Mask|视频-触觉路由解耦]]和[[Touch-Aware Bias|触觉感知动作偏置]]
- **输出**: 未来视觉隐变量 $z^v_{t+1:t+T}$ + 未来触觉接触状态 $z^\tau_{t+1:t+K}$ + [[Action Chunking|动作块]] $a_{t:t+H-1}$
- **联合 token 序列**: $X = [X^v \mid X^\tau \mid X^a \mid X^s]$（视觉 token、未来触觉寄存器、动作 token、上下文 token）

未来触觉寄存器（tactile registers）与动作预测时间范围对齐，使动作 token 能在生成修正动作的具体时间步读取预测的接触状态。

推理时，未来触觉状态和动作均从噪声初始化并联合去噪；真实未来触觉状态仅在训练时用作监督目标。

### 核心模块

#### 模块 1: [[VideoClean Attention Mask|VideoClean 注意力掩码]]

**设计动机**: 朴素的视触觉 WAM 允许所有 token 组互相关注，包括视频 query 访问触觉 key/value，这种对称注意力模式会导致"触觉污染"——稀疏、局部的触觉事件对未来视频弱预测，却会直接扰动预训练的视觉动力学路径。

**具体实现**:
- 阻断视频 query 对触觉 key/value 的访问（设置注意力 logit 为 $-\infty$）
- 保留动作 query 对触觉 token 的访问，确保触觉信息仍可用于动作生成
- 该掩码作用方向是单向的：只阻断 "video→tactile"，不阻断 "action→tactile"

#### 模块 2: [[Touch-Aware Bias|触觉感知偏置]] 与 [[Touch-State Proxy|触觉状态/变化代理]]

**设计动机**: 让触觉隐状态对动作生成"有意义"，需要用运动衍生的代理目标约束触觉表征，而不是依赖未标定的力传感器标签。

**具体实现**:
- 触觉状态代理 $c^\tau$：捕捉局部接触负载（contact loading）
- 触觉变化代理 $\Delta c^\tau$：突出阶跃式接触转变（首次接触、滑动开始、压缩变化、卡阻）
- 两者均通过触觉图像运动衍生，**不是**标定力标签，也**不是**额外的观测 token，而是辅助监督目标
- 触觉感知偏置 $b^\tau_i$ 由触觉变化代理映射得到（[[阈值饱和函数|thresholded saturating function]] $g(\cdot)$），仅叠加到 "action-query → tactile-key" 的注意力 logit 上
- 训练时偏置由目标触觉变化代理通过 [[teacher forcing|教师强制]]构造；推理第一步无偏置（$b^{\tau,(0)}=0$），之后步骤使用上一步预测的触觉变化代理构造偏置

#### 模块 3: 动作时间范围对齐的触觉寄存器（[[Action-Horizon-Aligned Tactile Registers]]）

**设计动机**: 让触觉表征压缩为固定数量的紧凑 token，同时保留时间-锚点-传感器-槛位的对齐信息。

**具体实现**:
- 左右两个视觉触觉传感器的图像帧输入冻结的触觉表征模型（可用 [[UniT]] 风格的连续触觉隐变量）
- 交叉注意力 token 适配器将每个 anchor-sensor 对的 patch 级触觉特征压缩为固定数量的紧凑 token：先投影到模型维度，再用可学习的 slot token 通过交叉注意力查询，接残差 [[MLP]]
- 每个动作块使用 $A$ 个触觉锚点，每个锚点含 $S=2$ 个触觉传感器（左右）

---

## 关键公式

### 公式 1: [[World Action Model|视觉 WAM 基线]]分布

$$
p_{\theta}\left(z^{v}_{t+1:t+T},\,a_{t:t+H-1}\mid o^{v}_{\leq t},s_{\leq t},l\right)
$$

**含义**: 给定视觉历史、本体状态、语言指令，联合预测未来视觉隐变量和动作块——这是不含触觉模态的基线形式。

**符号说明**:
- $z^v_{t+1:t+T}$: 未来 $T$ 步的视觉隐变量
- $a_{t:t+H-1}$: 动作范围 $H$ 的动作块
- $o^v_{\le t}, s_{\le t}, l$: 视觉历史、本体状态历史、语言指令

### 公式 2: [[World Action Model|Tactile-WAM 联合分布]]

$$
p_{\theta}\left(z^{v}_{t+1:t+T},\,z^{\tau}_{t+1:t+K},\,a_{t:t+H-1}\mid o^{v}_{\leq t},o^{\tau}_{\leq t},s_{\leq t},l\right)
$$

**含义**: 在视觉 WAM 基础上加入触觉历史观测，并联合预测未来触觉接触状态——这是 Tactile-WAM 的核心建模目标。

**符号说明**:
- $z^\tau_{t+1:t+K}$: 未来 $K$ 步的触觉接触状态隐变量
- $o^\tau_{\le t}$: 触觉历史观测

### 公式 3: [[Diffusion Transformer|联合 Token 序列]]构造

$$
X = [X^{v} \mid X^{\tau} \mid X^{a} \mid X^{s}]
$$

**含义**: 视觉、触觉、动作、上下文（状态+语言）四类 token 拼接为单一序列，输入共享的去噪骨干。

**符号说明**:
- $X^v, X^\tau, X^a, X^s$: 视觉 token、未来触觉寄存器、动作 token、上下文 token

### 公式 4: [[VideoClean Attention Mask|VideoClean]] 掩码

$$
M^{\mathrm{vc}}_{q,k}=
\begin{cases}
-\infty, & G(q)=V,\; G(k)=\tau, \\
0, & \text{otherwise},
\end{cases}
$$

**含义**: 当 query 属于视频 token 组、key 属于触觉 token 组时，掩码值为 $-\infty$（屏蔽该注意力路径），其他情况不影响。

**符号说明**:
- $G(\cdot)$: 返回 token 所属组的函数
- $V$: 视频 token 组标记；$\tau$: 触觉 token 组标记
- $q, k$: query token、key token 的索引

### 公式 5: 基础注意力偏置

$$
\bar{B}_{q,k}=B^{0}_{q,k}+M^{\mathrm{vc}}_{q,k}
$$

**含义**: 在骨干网络原有的因果/分块/局部注意力规则上叠加 VideoClean 掩码，得到屏蔽后的基础偏置。

**符号说明**:
- $B^0_{q,k}$: 骨干网络原始的注意力偏置（因果/分块/局部规则）
- $\bar{B}_{q,k}$: 叠加 VideoClean 掩码后的基础偏置

### 公式 6: [[Touch-Aware Bias|触觉感知偏置]]映射

$$
b_{i}^{\tau}=g(\delta_{i}^{\tau})
$$

**含义**: 第 $i$ 个触觉锚点的触觉变化信号 $\delta_i^\tau$（训练时为目标触觉变化代理，推理时为上一步预测值）经函数 $g(\cdot)$ 映射为标量偏置。

**符号说明**:
- $\delta_i^\tau$: 锚点 $i$ 的触觉变化信号
- $g(\cdot)$: 阈值饱和函数（详见公式 14-15）
- $b_i^\tau$: 映射得到的标量偏置

### 公式 7: 完整注意力 Logit（含触觉感知偏置）

$$
B_{q,k}=\bar{B}_{q,k}+\mathbb{I}[G(q)=A,\;G(k)=\tau]\,b^{\tau}_{a(k)}
$$

**含义**: 仅当 query 属于动作 token 组、key 属于触觉 token 组时，叠加该触觉锚点对应的偏置；不缩放触觉特征本身，不增加其他 token 组对触觉的注意力。

**符号说明**:
- $A$: 动作 token 组标记
- $a(k)$: 触觉 key $k$ 所属的触觉锚点索引
- $\mathbb{I}[\cdot]$: 指示函数

### 公式 8: 触觉 Token 总数

$$
L_{\tau}=ASQ
$$

**含义**: 触觉 token 总数由锚点数、传感器数、每锚点-传感器对的紧凑 token 数相乘得到。

**符号说明**:
- $A$: 每个动作块的触觉锚点数
- $S=2$: 每锚点的传感器数（左右）
- $Q$: 交叉注意力适配器输出的 slot token 数

### 公式 9: [[Action-Horizon-Aligned Tactile Registers|触觉寄存器]]构造

$$
r^{\tau}_{i,s,q}=W_{\tau}\tilde{z}^{\tau}_{i,s,q}+e^{time}_{i}+e^{anchor}_{i}+e^{sensor}_{s}+e^{slot}_{q}
$$

**含义**: 触觉寄存器由当前去噪步的噪声触觉 token 线性投影后，叠加时间、锚点、传感器、槛位四类位置编码构成。

**符号说明**:
- $\tilde{z}^\tau_{i,s,q}$: 锚点 $i$、传感器 $s$、槛位 $q$ 的噪声触觉 token
- $W_\tau$: 线性投影矩阵
- $e^{time}_i, e^{anchor}_i, e^{sensor}_s, e^{slot}_q$: 对应的位置/身份嵌入

### 公式 10: [[Touch-State Proxy|触觉变化代理]]定义

$$
\Delta c_{i}^{\tau}=c_{i}^{\tau}-c_{i-1}^{\tau}
$$

**含义**: 触觉变化代理是相邻时刻触觉状态代理的差分，用于捕捉接触转变（不是绝对接触量级，而是变化量）。

**符号说明**:
- $c_i^\tau$: 第 $i$ 步的触觉状态代理（衡量局部接触负载）
- $\Delta c_i^\tau$: 第 $i$ 步相对上一步的触觉变化代理

### 公式 11: 触觉代理头预测

$$
\left(\hat{u}^{\tau}_{i},\hat{c}^{\tau}_{i},\widehat{\Delta c^{\tau}}_{i}\right)=D_{\phi}(h^{\tau}_{i})
$$

**含义**: 一个轻量代理头 $D_\phi$ 从触觉隐状态预测触觉速度、触觉状态代理、触觉变化代理三个量。

**符号说明**:
- $h_i^\tau$: 锚点 $i$ 的触觉隐状态
- $\hat{u}_i^\tau$: 预测的触觉速度（用于去噪）
- $\hat{c}_i^\tau, \widehat{\Delta c^\tau}_i$: 预测的触觉状态代理、触觉变化代理
- $D_\phi$: 代理头网络（参数 $\phi$）

### 公式 12: 代理监督损失

$$
\begin{aligned}
\mathcal{L}_{\mathrm{state}} &= \mathrm{SmoothL1}\left(\hat{c}^{\tau},c^{\tau,\star}\right), \\
\mathcal{L}_{\mathrm{change}} &= \mathrm{SmoothL1}\left(\widehat{\Delta c^{\tau}},\Delta c^{\tau,\star}\right).
\end{aligned}
$$

**含义**: 用 [[SmoothL1 Loss]] 分别监督触觉状态代理预测和触觉变化代理预测，对齐真实运动衍生目标。

**符号说明**:
- $c^{\tau,\star}, \Delta c^{\tau,\star}$: 真实（运动衍生）的触觉状态代理、触觉变化代理目标
- $\mathrm{SmoothL1}$: 平滑 L1 损失，对小误差为二次、大误差为线性，鲁棒于异常值

### 公式 13: 触觉变化幅度

$$
d_{i}=\left\|\Delta c^{\tau}_{i}\right\|_{2}
$$

**含义**: 取触觉变化代理的 L2 范数作为该锚点的变化幅度标量。

**符号说明**:
- $d_i$: 第 $i$ 个锚点的触觉变化幅度

### 公式 14-15: [[Touch-Aware Bias|触觉感知偏置]]的阈值饱和构造

$$
s_{i}=\mathrm{clip}_{[0,1]}\left[\tanh\left(\frac{\mathrm{ReLU}(d_{i}-\theta)}{T_{c}}\right)\right]
$$

$$
b^{\tau}_{i}=\mathrm{clip}\left(\alpha s_{i},\,0,\,b_{\max}\right)
$$

**含义**: 先用阈值 $\theta$ 过滤小幅噪声变化，再用 $\tanh$ 饱和并裁剪到 $[0,1]$ 得到变化分数 $s_i$；再线性缩放并裁剪到 $[0, b_{\max}]$ 得到最终加性偏置。

**符号说明**:
- $\theta$: 变化幅度阈值（低于该值视为无显著变化）
- $T_c$: 温度系数，控制饱和陡峭程度
- $\alpha$: 缩放系数；$b_{\max}$: 偏置上界
- 当 $b_i^\tau=2$ 时，未归一化 softmax 权重等价于乘以 $\exp(2)$

### 公式 16: 完整注意力 Logit（附录展开形式）

$$
B_{q,k}=\bar{B}_{q,k}+\mathbb{I}[G(q)=A,\;G(k)=\tau]\,b^{\tau}_{a(k)}
$$

**含义**: 与公式 7 等价，附录中重述以衔接公式 13-15 的偏置构造细节。

**符号说明**: 同公式 7。

### 公式 17: 训练时的插值噪声

$$
\tilde{z}^{\tau}_{\lambda}=(1-\lambda)z^{\tau}_{0}+\lambda\epsilon^{\tau}, \qquad \tilde{a}_{\lambda}=(1-\lambda)a_{0}+\lambda\epsilon^{a}
$$

**含义**: 干净触觉 token 和动作块分别与高斯噪声按插值系数 $\lambda$ 混合，用于构造去噪训练样本。

**符号说明**:
- $z_0^\tau, a_0$: 干净的触觉 token、动作块
- $\epsilon^\tau, \epsilon^a$: 标准高斯噪声
- $\lambda$: 插值/噪声系数

### 公式 18: 总训练损失

$$
\mathcal{L}=\mathcal{L}_{\mathrm{video}}+\lambda_{a}\mathcal{L}_{\mathrm{action}}+\lambda_{\tau}\mathcal{L}_{\mathrm{tactile}}+\lambda_{s}\mathcal{L}_{\mathrm{state}}+\lambda_{c}\mathcal{L}_{\mathrm{change}}
$$

**含义**: 视频去噪损失、动作去噪损失、触觉去噪损失、触觉状态代理损失、触觉变化代理损失的加权和，联合监督多模态去噪与代理预测。

**符号说明**:
- $\mathcal{L}_{video}, \mathcal{L}_{action}, \mathcal{L}_{tactile}$: 视频/动作/触觉的去噪损失
- $\lambda_a, \lambda_\tau, \lambda_s, \lambda_c$: 对应损失的权重系数

### 公式 19-20: 推理时触觉感知偏置的逐步构造

$$
b_{i}^{\tau,(0)}=0
$$

$$
b_{i}^{\tau,(m)}=g\left(\widehat{\Delta c^{\tau}}^{(m-1)}_{i}\right),\quad m\geq 1
$$

**含义**: 推理第一步（$m=0$）无法获得任何预测的触觉变化代理，因此偏置为零；从第二步起，用上一步（$m-1$）预测的触觉变化代理构造偏置，避免在测试时使用不可得的未来真值标签。

**符号说明**:
- $m$: 去噪步序号
- $\widehat{\Delta c^\tau}^{(m-1)}_i$: 第 $m-1$ 步预测的触觉变化代理

### 公式 21: 总体成功率指标

$$
\mathrm{Success}_{\mathrm{overall}}=\frac{\sum_{\tau\in\mathcal{T}}N_{\mathrm{succ}}(\tau)}{\sum_{\tau\in\mathcal{T}}N_{\mathrm{eval}}(\tau)}
$$

**含义**: 在全部仿真任务集合 $\mathcal{T}$ 上，成功 rollout 总数除以评估 rollout 总数，得到总体成功率。

**符号说明**:
- $\mathcal{T}$: 全部仿真任务集合（9 个 ManiFeel 任务）
- $N_{succ}(\tau), N_{eval}(\tau)$: 任务 $\tau$ 的成功次数、评估总次数

---

## 关键图表

### Figure 1: 总览与三种接触事件示意

![[Tactile-WAM_fig1_overview.png]]

**说明**: 该图为论文标题页 teaser 图（仅出现在 PDF 排版中，arXiv HTML 版本未渲染该图，已从 PDF 中提取保存至本地）。左上角展示 RGB-only WAM 的局限——视觉上合理的未来预测可能遗漏滑动（Slip）、卡阻（Jamming）、对齐误差（Misalignment）等接触状态；中上展示朴素视触觉 WAM（Naive VT-WAM）的"触觉污染"现象——触觉 token 自由参与注意力后，预测的视频帧出现明显噪声伪影；右上展示 Tactile-WAM 的非对称路由（Asymmetric Routing）如何保持视频预测干净。中部为模型整体流程图（与 Figure 2 一致但更精简），并展示仿真任务（Bolt Nut、Bulb Insert、Gear Assemble、Power Insert）与真机任务对照。下方三个面板分别展示 Aligned（对齐/稳定接触）、Slip（滑动）、Jamming（卡阻）三种事件下 RGB、触觉、力、$\Delta$ 力随时间变化的模式，以及对应的偏置策略（keep/correct/safety）和动作决策（continue/retract/back off）。

### Figure 2: 方法总览（Method Overview）

![Figure 2](https://arxiv.org/html/2606.26663v1/figures/fig2.png)

**说明**: 完整方法流程图，分四个阶段：(1) 观测上下文——RGB 历史、左右触觉历史、机器人状态、语言指令编码为 token；(2) 单一 Wan-based WAM 去噪骨干，对视觉/触觉/动作 token 联合去噪；(3) [[Tactile Asymmetric Attention Mechanism (TAAM)|Tactile Asymmetric Attention]]——VideoClean 表格直观展示视频 query 对触觉 key 的访问被阻断（红叉），而动作 query 对触觉 key 的访问被保留并叠加偏置（绿勾 +bias），偏置由触觉隐状态经轻量 MLP 代理头预测虚拟力 $f_{vf}$ 及其变化量 $\Delta f_{vf}$，再经范数+阈值+tanh 映射为加性偏置；(4) 联合输出——未来视频、未来触觉接触状态、每指接触轨迹（contact trace）、动作块。图中术语"virtual force"对应正文中的触觉状态/变化代理 $c^\tau/\Delta c^\tau$。

### Figure 3: 主要结果（Main Results）

![Figure 3](https://arxiv.org/html/2606.26663v1/figures/fig3_arxiv.png)

**说明**: (a) 仿真主结果：Tactile-WAM 在接触为主的插入装配任务（螺栓螺母装配、齿轮插入、电源插入、USB 插入）上大幅超越 DreamZero 和 π₀.₅；(b) 真机结果：五个接触密集任务上 Tactile-WAM 整体成功率 51%，显著高于 RGB-only DreamZero（18%）和 π₀.₅ 基线。完整逐任务数据见附录 Table III。

### Figure 4: 真机实验平台

![Figure 4](https://arxiv.org/html/2606.26663v1/figures/fig_real_setup.png)

**说明**: 真机评估覆盖五个接触密集任务：插销、螺母穿线、齿轮啮合、灯泡插入、电源插入。平台使用 xArm 机械臂配备 Tac-UMI 风格夹爪，外部 RealSense L515 传感，夹爪安装鱼眼相机/触觉观测模块。任务夹具包括插头插座、螺母螺纹、齿轮啮合、灯泡、销孔插座装配体。

### Figure 5: 仿真任务定性过程可视化（第一部分）

![Figure 5-1](https://arxiv.org/html/2606.26663v1/figures/sim_figs/1.png)
![Figure 5-2](https://arxiv.org/html/2606.26663v1/figures/sim_figs/2.png)
![Figure 5-3](https://arxiv.org/html/2606.26663v1/figures/sim_figs/3.png)
![Figure 5-4](https://arxiv.org/html/2606.26663v1/figures/sim_figs/4.png)

**说明**: 每个面板展示任务执行过程中的多视角 RGB 观测和左右触觉观测随时间演变，覆盖 ManiFeel 仿真任务的前半部分。

### Figure 6: 仿真任务定性过程可视化（第二部分）

![Figure 6-1](https://arxiv.org/html/2606.26663v1/figures/sim_figs/5.png)
![Figure 6-2](https://arxiv.org/html/2606.26663v1/figures/sim_figs/6.png)
![Figure 6-3](https://arxiv.org/html/2606.26663v1/figures/sim_figs/7.png)
![Figure 6-4](https://arxiv.org/html/2606.26663v1/figures/sim_figs/8.png)

**说明**: 进一步展示视觉和触觉观测流在接触密集型操作中的应用，覆盖剩余仿真任务。

### Figure 7: 真机定性过程可视化（第一部分）

![Figure 7-1](https://arxiv.org/html/2606.26663v1/figures/real_figs/1.png)
![Figure 7-2](https://arxiv.org/html/2606.26663v1/figures/real_figs/2.png)
![Figure 7-3](https://arxiv.org/html/2606.26663v1/figures/real_figs/3.png)
![Figure 7-4](https://arxiv.org/html/2606.26663v1/figures/real_figs/4.png)

**说明**: 每个面板展示真机执行序列随时间变化，包括夹爪安装 RGB 观测、左右视触觉观测、左右 $\Delta$ 力热力图，展现局部接触信号在抓取、对齐、穿线、插入、完成各阶段的演变。

### Figure 8: 真机定性过程可视化（第二部分）

![Figure 8-1](https://arxiv.org/html/2606.26663v1/figures/real_figs/5.png)
![Figure 8-2](https://arxiv.org/html/2606.26663v1/figures/real_figs/6.png)
![Figure 8-3](https://arxiv.org/html/2606.26663v1/figures/real_figs/7.png)
![Figure 8-4](https://arxiv.org/html/2606.26663v1/figures/real_figs/8.png)

**说明**: 进一步展示真实世界接触密集操作中视觉、触觉、$\Delta$ 力观测的时间演化。

### Figure 9: VideoClean 定性对比

![Figure 9-1](https://arxiv.org/html/2606.26663v1/figures/clean_figs/1.png)
![Figure 9-2](https://arxiv.org/html/2606.26663v1/figures/clean_figs/2.png)
![Figure 9-3](https://arxiv.org/html/2606.26663v1/figures/clean_figs/3.png)

**说明**: 对比非清洁视触觉 WAM（non-clean）与 VideoClean 变体在同一真机序列上解码出的 RGB 预测。红框标出触觉污染引入视觉伪影或几何畸变的区域；VideoClean 变体更好地保留了夹爪和物体的结构完整性。

---

### Table I: VideoClean 的 RGB 预测质量对比（5.5K checkpoint）

| Metric | Non-clean | VideoClean | Change |
|--------|-----------|-----------|---------|
| RGB MSE ↓ | 0.01524 | 0.01457 | -4.42% |
| RGB MAE ↓ | 0.05813 | 0.05732 | -1.39% |
| PSNR ↑ | 18.91 | 19.05 | +0.14 dB |
| Global SSIM ↑ | 0.9054 | 0.9072 | +0.0017 |

**说明**: 使用 9 个任务的 108 个任务均衡 ManiFeel 样本，在 4-chunk 预测范围下评估。VideoClean 在全部四个指标上一致提升解码 RGB 预测质量，支持"触觉污染"假说——阻断视频 query 对触觉 key/value 的访问有助于保护视触觉 WAM 训练中的视觉预测路径。

### Table II: Tactile-WAM 组件消融分析

| Method | Future Touch | VideoClean | Proxies & Bias | Success |
|--------|-------------|-----------|-----------------|---------|
| RGB-only WAM | ✗ | ✗ | ✗ | 0.058 |
| Naive VT-WAM | ✓ | ✗ | ✗ | 0.013 |
| + VideoClean | ✓ | ✓ | ✗ | 0.049 |
| Full Tactile-WAM | ✓ | ✓ | ✓ | 0.447 |

**关键发现**: 朴素加入未来触觉 token（Naive VT-WAM）反而把成功率从 5.8% 降到 1.3%，证明无约束路由的触觉输入会损害视觉动力学路径；加入 VideoClean 恢复到 4.9%，但仍接近 RGB-only 基线，说明"保护视频预测"是必要但不充分条件；只有同时加入触觉状态/变化代理与触觉感知偏置（完整 Tactile-WAM），成功率才跃升至 44.7%。

### Table III: ManiFeel 完整仿真结果

| Task | LeRobot π₀.₅ | DreamZero | Tactile-WAM |
|------|------|------|------|
| ball sorting | 0/50 (0.0%) | 23/50 (46.0%) | 26/50 (52.0%) |
| bolt-nut | 0/50 (0.0%) | 3/50 (6.0%) | 36/50 (72.0%) |
| bulb insertion | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| gear insertion | 0/50 (0.0%) | 0/50 (0.0%) | 50/50 (100.0%) |
| object search | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| peg insertion | 0/50 (0.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| peg reorientation | 6/50 (12.0%) | 0/50 (0.0%) | 0/50 (0.0%) |
| power insertion | 0/50 (0.0%) | 0/50 (0.0%) | 39/50 (78.0%) |
| USB insertion | 0/50 (0.0%) | 0/50 (0.0%) | 50/50 (100.0%) |
| **Overall** | 6/450 (1.3%) | 26/450 (5.8%) | 201/450 (44.7%) |
| **Contact-centric subset** | 0/200 (0.0%) | 3/200 (1.5%) | 175/200 (87.5%) |

**说明**: 每格为 50 次评估中的成功次数。接触为主子集（contact-centric subset）取螺栓螺母装配、齿轮插入、电源插入、USB 插入四个任务的平均。可见 Tactile-WAM 的优势高度集中在接触为主任务（87.5% vs DreamZero 1.5%），但在 bulb insertion、object search、peg insertion、peg reorientation 上三种方法均接近 0%，说明触觉感知 WAM 并不能单独解决视觉搜索、探索或长时程操作问题。

---

## 实验

### 数据集 / 基准

| 名称 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[ManiFeel]] 仿真基准 | 9 个任务 × 50 rollouts | 涵盖插入、装配、重定向、搜索、分类的视触觉操作基准 | 仿真训练/评估 |
| 真机基准（自建） | 5 个接触密集任务 × 100 trials | xArm + Tac-UMI 夹爪 + RealSense L515 + 鱼眼/触觉模块 | 真机评估 |

### 实现细节

- **Backbone**: [[Wan]] 视频生成模型改造的因果 [[Diffusion Transformer|DiT]]（Wan/WAM 兼容去噪骨干）
- **触觉表征**: 冻结的 [[UniT]] 风格连续触觉隐变量编码器，或直接加载预计算的冻结触觉特征
- **触觉适配器**: 交叉注意力 token 适配器 + 残差 MLP，将 patch 级触觉特征压缩为固定数量紧凑 token
- **基线对比**: [[pi0.5|π₀.₅]] 通用动作策略基线、基于 [[DreamZero]] 的 RGB-only WAM 基线
- **评估协议**: 仿真每任务 50 rollouts；真机每任务在匹配协议下统一评估，所有方法共享观测流、动作接口、任务夹具

### 可视化结果

- VideoClean 定性对比（Figure 9）显示触觉污染会在预测的夹爪/物体区域引入可见伪影和几何畸变，VideoClean 变体显著改善
- Aligned/Slip/Jamming 三种接触事件（Figure 1 下方面板）对应不同的触觉感知偏置策略（keep/correct/safety）与动作决策（continue/retract/back off），体现触觉变化代理对动作修正的指导作用

---

## 批判性思考

### 优点

1. **问题诊断精准**: "触觉污染"这一失败模式的识别和量化（Table I/II）非常清晰，直接证明了对称注意力是视触觉 WAM 性能下降的关键原因，而非简单归因于"触觉无用"
2. **设计极简且无需额外学习参数路由结构**: VideoClean 是纯粹的注意力掩码规则（无学习参数），触觉感知偏置只用轻量 MLP 代理头+阈值饱和函数，整体侵入性很小，便于在预训练视频 WAM 骨干上快速适配
3. **代理目标设计务实**: 触觉状态/变化代理不依赖标定力传感器，仅从触觉图像运动衍生，降低了对昂贵力/力矩传感器标定数据的依赖

### 局限性

1. **依赖时间对齐的多模态流**: 假设 RGB、本体状态、动作、触觉时间严格对齐，在更快速的灵巧操作中，接触事件可能比策略时间范围更高频发生（论文 Appendix D 自述）
2. **触觉监督非标定力信号**: 触觉状态/变化代理是运动衍生的代理量，而非真实力/力矩测量，部署到论文基准之外的场景时需要额外验证传感器延迟、掉帧、标定漂移、动作-触觉错位等问题
3. **保守路由先验的可扩展性存疑**: VideoClean 和 TAAM 是为视觉预训练 WAM 骨干设计的保守路由先验，论文自己承认：当触觉形变在 RGB 中直接可见，或在大规模配对视触觉 rollout 上训练时，可学习/自适应的跨模态路由可能更优
4. **部分任务类型仍接近零成功率**: object search、peg insertion、peg reorientation、bulb insertion 在三种方法下均接近 0%，说明该方法主要解决"接触敏感修正"问题，而非视觉搜索、探索或长时程恢复问题

### 潜在改进方向

1. **自适应/可学习的跨模态路由**: 替代固定的 VideoClean 掩码规则，让模型根据任务/模态可见性自动决定路由强度
2. **更密集或事件触发的触觉采样**: 解决高频接触事件（如快速灵巧操作）与策略时间范围不匹配的问题
3. **结合标定力/力矩信号**: 在触觉运动代理基础上融合真实力测量，提升触觉感知偏置的物理可解释性
4. **扩展到长时程恢复任务**: 论文中搜索、重定向等任务表现仍差，需要结合更强的视觉探索或恢复策略

### 可复现性评估

- [ ] 代码开源（未发现 GitHub 仓库或项目主页）
- [ ] 预训练模型（未提及权重发布）
- [x] 训练细节完整（损失函数、训练/推理流程、超参数符号在附录 B/C 中详尽列出）
- [x] 数据集可获取（[[ManiFeel]] 是公开基准；真机数据为自建，未提及是否公开）

---

## 关联笔记

### 基于

- [[World Action Model]]: Tactile-WAM 延续 WAM 范式（联合预测未来状态与动作），并将其扩展到触觉模态
- [[DreamZero]]: 视觉 WAM 基线，Tactile-WAM 沿用其"适配视频生成骨干联合预测视觉观测与动作块"的思路
- [[ManiFeel]]: 仿真评估基准，提供 RGB-only WAM、触觉条件非 WAM 策略、物理感知 WAM 之间的统一对比环境

### 对比

- [[Dream-Tac]]: 最直接相关的同类工作，同样提出"统一触觉 World Action Model"，但用接触门控自注意力（CASA，门控值由触觉帧差自动计算）而非 Tactile-WAM 的"非对称掩码+预测变化偏置"路线；两者都强调触觉应被选择性路由而非无约束融合
- [[pi0.5|π₀.₅]]: 通用视觉-语言-动作策略基线，无触觉模态、无未来预测，在接触密集任务上表现最弱（仿真 1.3%）
- [[OmniVTA]] / [[VTAM]] / [[DreamTacVLA]] / [[TacForeSight]]: 同期相关视触觉世界建模/触觉前瞻工作，论文在 Related Work 中明确将自身定位为"触觉作为生成式物理状态，经非对称动作路径路由"这一更精确的架构问题，而非笼统声称"首个触觉 WAM"

### 方法相关

- [[Tactile Asymmetric Attention Mechanism (TAAM)]]: 本文核心创新，组合 VideoClean 与触觉感知偏置
- [[VideoClean Attention Mask]]: 阻断视频 query 对触觉 key/value 访问的注意力掩码
- [[Touch-Aware Bias]]: 由触觉变化代理驱动、仅作用于动作-触觉注意力路径的加性偏置
- [[Touch-State Proxy]]: 运动衍生的触觉状态/变化代理目标，替代标定力标签
- [[Action Chunking]]: 动作预测的基本单元
- [[teacher forcing|Teacher Forcing]]: 训练时触觉感知偏置使用真实目标触觉变化代理构造的设计
- [[SmoothL1 Loss]]: 触觉状态/变化代理的监督损失函数
- [[UniT]]: 可选的触觉表征编码器
- [[Diffusion Transformer|DiT]]: 联合去噪骨干的基础架构

### 硬件/数据相关

- [[ManiFeel]]: 仿真基准，9 个任务
- Tac-UMI 风格夹爪、RealSense L515：真机实验平台关键硬件（暂无独立概念笔记）

---

## 速查卡片

> [!summary] Tactile-WAM
> - **核心**: 提出 TAAM 非对称注意力，用 VideoClean 掩码隔绝触觉对视频预测的污染，用触觉变化偏置选择性引导动作去噪
> - **方法**: 单一 Wan-based DiT 联合去噪视觉/触觉/动作 token；VideoClean 阻断 video→tactile 注意力但保留 action→tactile；触觉状态/变化代理（运动衍生，非标定力）驱动加性偏置仅作用于 action-query→tactile-key
> - **结果**: ManiFeel 仿真整体成功率 44.7%（vs DreamZero 5.8%，π₀.₅ 1.3%），接触为主子集 87.5%；真机 5 任务整体 51%（vs DreamZero 18%）
> - **代码**: 未开源/未发现项目主页

---

*笔记创建时间: 2026-06-27*
