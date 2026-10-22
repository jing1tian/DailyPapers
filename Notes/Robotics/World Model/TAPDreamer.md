---
title: "TAPDreamer: Transferable Adversarial Patches for World Action Models"
method_name: "TAPDreamer"
authors: [Xuanyu Lu, Fengqing Jiang, Kaiyuan Zheng, Yichen Feng, Yaorui Ding, Yuetai Li, Zhen Xiang, Bhaskar Ramasubramanian, Basel Alomair, Luyao Niu, Radha Poovendran]
year: 2026
venue: arXiv
tags: [adversarial-attack, world-action-model, visual-encoder, transferable-patch, security, embodied-ai, robustness]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2610.06814
created: 2026-10-22
---

# 论文笔记：TAPDreamer: Transferable Adversarial Patches for World Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of Washington, University of Georgia, Western Washington University, King Abdulaziz City for Science and Technology, HUMAIN |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[FastWAM]]、[[DreamWAM]]、[[Motus]]、随机同尺寸补丁 |
| 链接 | [arXiv](https://arxiv.org/abs/2610.06814) |

---

## 一句话总结

> TAPDreamer 仅利用公开[[Visual Encoder|视觉编码器]]（无需访问目标策略）优化一块固定局部[[对抗补丁攻击|对抗补丁]]，即可将 [[FastWAM]] 在 40 个 [[LIBERO]] 任务和 50 个 [[RoboTwin]] 任务上的成功率从 ~97% 和 ~91% 均降至 0%，且同一补丁可迁移攻击多种 [[WAM]] 变体。

---

## 核心贡献

1. **无查询可迁移补丁攻击**：TAPDreamer 仅需白盒访问一个公开[[Visual Encoder|视觉编码器]]和少量源任务帧，无需访问目标策略权重、动作标签、奖励信号或仿真器，即可制作高效的可迁移[[对抗补丁攻击|对抗补丁]]。
2. **揭示共享视觉编码器攻击面**：识别出 WAM 使用公共预训练[[Visual Encoder|视觉编码器]]这一关键暴露面——补丁通过编码器潜变量扰动破坏策略，而非直接攻击动作生成模块。
3. **注意力传播机制分析**：通过公式化分解证明，局部补丁引起的潜变量变化通过[[Attention Mechanism|注意力机制]]的瓶颈扩散至全部空间位置，并在不同观测之间形成共享的"源模板"分量，这是可迁移性的内在机制。
4. **跨模型跨任务可迁移性**：在 [[FastWAM]]、[[DreamWAM]]-uncond、DreamWAM-joint 和 [[Motus]] 上验证成功，并在真实 SO-101 机器人上取得验证。

---

## 问题背景

### 要解决的问题

[[WAM|World Action Model (WAM)]] 通过视觉观测闭环控制机器人，其摄像头输入和共享[[Visual Encoder|视觉编码器]]形成一个暴露的攻击通道。现有研究表明 VLA 策略可被对抗攻击破坏，但这些攻击通常需要目标模型的梯度访问或实时查询，在 WAM 场景下不切实际。

### 现有方法的局限

- 基于动作输出查询的攻击（如 [[BadWAM]]）需要实时查询目标模型，每步多次调用策略
- 白盒补丁攻击（如 [[UPA]]、[[UADA]]）直接针对 VLA，迁移效果弱（成功率仅降至 75–76%）
- 现有[[对抗补丁攻击|对抗补丁]]方法未充分利用 WAM 中公开共享的视觉编码器作为迁移桥梁

### 本文的动机

WAM 普遍依赖公开的大规模预训练视觉编码器（如 DINO-V2、SigLIP），攻击者可在本地利用该编码器优化补丁——**编码器潜变量的大位移必然传导至策略失败**。Figure 1 展示了同一个离线优化的补丁在多个不同 WAM 和任务上复用。

---

## 方法详解

### 威胁模型

TAPDreamer 在**纯测试时**（无训练时访问）设定下运行：

- **可访问**：公开[[Visual Encoder|视觉编码器]]（白盒，含权重和梯度）；源任务的少量干净帧（$J=6$）
- **不可访问**：目标 WAM 的权重、动作输出、奖励信号、仿真器
- **扰动约束**：固定二值掩码 $M$ 内（约 6.5% 的输入像素）替换图像内容（Eq. 3）
- **不修改**：动作、状态、语言指令或本体感知信号

### 补丁优化流程

TAPDreamer 分两阶段：**离线补丁优化** + **固定补丁部署**（对应 Figure 1 的 (a) 和 (b)）：

1. 从源任务采集 $J$ 帧干净观测 $\{x_j\}_{j=1}^J$
2. 通过冻结[[Visual Encoder|视觉编码器]] $E$ 缓存干净潜变量 $Z_j^0 = E(x_j)$
3. 用 [[Adam优化器|Adam]] 最大化 L1 潜变量位移，迭代 500 步（Algorithm 1）
4. 将最优补丁 $p^*$ 贴在固定掩码位置，无需修改即可部署到任意任务和 WAM 变体

### 注意力传播机制

Figure 4 展示了局部变化如何通过[[Attention Mechanism|注意力]]瓶颈扩散至全部位置（Figure 5 可视化）：

- 补丁仅扰动掩码区域像素，但经 [[VAE]] 编码后，注意力权重 $A$ 和值矩阵 $V$ 的联合变化将扰动信号广播到**全部空间位置**
- 不同观测帧之间的 off-footprint 变化方向高度一致（余弦相似度 0.981），形成**共享源模板** $\Delta\bar{Z}_\text{src}$，这是补丁跨观测可迁移的数学根源

---

## 关键公式

### 公式 1: [[WAM|WAM 预测与动作生成]]

$$
\hat{Z}_{t+1:t+K_v} \sim W_\theta(\cdot \mid C_t), \quad a_{t:t+K_a-1} \sim A_\theta(\cdot \mid C_t,\, \hat{Z}_{t+1:t+K_v})
$$

**含义**: WAM 在上下文 $C_t$ 条件下先预测未来视觉潜变量序列，再据此生成动作块。

**符号说明**:
- $\hat{Z}_{t+1:t+K_v}$：未来 $K_v$ 步的预测视觉潜变量
- $C_t$：当前上下文（历史观测、语言指令等）
- $W_\theta$：世界预测分布，参数 $\theta$
- $A_\theta$：动作生成分布
- $K_v, K_a$：视觉预测和动作预测的时域长度

---

### 公式 2: [[WAM|闭环执行]]

$$
s_{t+K_e} = F(s_t,\, a_{t:t+K_e-1}), \quad o_{t+K_e} = O(s_{t+K_e})
$$

**含义**: 执行动作块后，环境状态和观测更新，形成与 WAM 的闭环。

**符号说明**:
- $s_t$：时刻 $t$ 的环境状态
- $F$：环境转移函数
- $O$：观测函数
- $K_e$：实际执行的动作步数（$1 \leq K_e \leq K_a$）

---

### 公式 3: [[对抗补丁攻击|补丁叠加]]

$$
T_p(x_t) = (1 - M) \odot x_t + M \odot p
$$

**含义**: 在固定二值掩码 $M$ 区域内用补丁 $p$ 替换原始图像像素，掩码外保持不变。

**符号说明**:
- $x_t$：原始输入图像
- $M$：固定二值掩码（约占输入 6.5%）
- $p$：待优化的对抗补丁
- $\odot$：逐元素乘法

---

### 公式 4: [[对抗补丁攻击|TAPDreamer 补丁优化目标]]

$$
\max_{p \in \mathcal{P}_\text{local}} \frac{1}{J \cdot N_Z} \sum_{j=1}^{J} \bigl\| E(T_p(x_j)) - E(x_j) \bigr\|_1
$$

**含义**: 在局部掩码约束下，最大化补丁帧与干净帧在编码器潜空间中的平均 L1 距离。

**符号说明**:
- $E$：冻结的公共[[Visual Encoder|视觉编码器]]
- $J$：优化所用帧数（论文中 $J=6$）
- $N_Z$：潜变量标量维度总数
- $\mathcal{P}_\text{local}$：掩码约束内的合法补丁集合

---

### 公式 5: [[Adam优化器|Algorithm 1 迭代损失]]

$$
\mathcal{L}_k = \frac{1}{J \cdot N_Z} \sum_{j=1}^{J} \bigl\| E(T_{p_k}(x_j)) - Z_j^0 \bigr\|_1, \quad Z_j^0 = E(x_j)
$$

**含义**: 第 $k$ 步迭代时，计算当前补丁 $p_k$ 相对于缓存干净潜变量的 L1 距离（最大化此损失）。

**符号说明**:
- $Z_j^0$：提前缓存的第 $j$ 帧干净潜变量
- $p_k$：第 $k$ 步的当前补丁参数

---

### 公式 6: 成功率

$$
\text{SR} = \frac{N_\text{success}}{N_\text{trials}} \times 100\%
$$

**含义**: 策略在一组试验中成功完成任务的百分比，是主要评估指标。

---

### 公式 7: [[Attention Mechanism|注意力输出变化分解]]

$$
\Delta H = \Delta U + \underbrace{(A \Delta V + \Delta A V + \Delta A \Delta V)}_{\text{注意力主导项}} W_O
$$

**含义**: 将注意力输出变化分解为残差项 $\Delta U$ 和注意力主导项，实验显示后者中注意力权重与值的联合变化 $\Delta A \Delta V$ 最大，是补丁扩散到 off-footprint 区域的核心机制。

**符号说明**:
- $H$：注意力输出（残差叠加后）
- $U$：残差输入
- $A$：注意力权重矩阵
- $V$：值矩阵
- $W_O$：输出投影矩阵
- $\Delta$：补丁扰动前后的差值

---

### 公式 8: [[Attention Mechanism|Off-footprint 变化的公共分量]]

$$
\Delta H_O = \mathbf{1} g^\top + R_H, \quad \|\Delta H_O\|_F \geq \sqrt{|O|} \cdot \|g\|_2 - \|R_H\|_F
$$

**含义**: Footprint 外所有位置的注意力变化可分解为公共向量 $g$ 和位置残差 $R_H$；$g$ 的存在给出可迁移位移量的下界，解释了为什么同一补丁在不同观测下均能产生显著扰动。

**符号说明**:
- $O$：footprint 外的位置集合
- $g$：所有位置的均值变化向量（公共分量）
- $R_H$：位置特定残差
- $\mathbf{1}$：全 1 向量

---

### 公式 9: [[对抗补丁攻击|源模板分解与迁移下界]]

$$
\Delta Z_i = \Delta\bar{Z}_\text{src} + e_i, \quad \|\Delta Z_i\|_1 \geq \|\Delta\bar{Z}_\text{src}\|_1 - \|e_i\|_1
$$

**含义**: 每个 held-out 观测的潜变量位移 $\Delta Z_i$ 可分解为源模板 $\Delta\bar{Z}_\text{src}$ 和个体残差 $e_i$，模板的大小提供了实测位移的下界，中位值保留率 0.833。

**符号说明**:
- $\Delta\bar{Z}_\text{src}$：源任务帧推导出的可迁移模板
- $e_i$：第 $i$ 个 held-out 观测的个体残差
- $\|\cdot\|_1$：L1 范数

---

### 公式 10: 有效秩（Effective Rank）

$$
\text{erank} = \exp\!\left(-\sum_j q_j \log q_j\right), \quad q_j = \frac{\sigma_j^2}{\sum_k \sigma_k^2}
$$

**含义**: 用于度量堆叠位移场矩阵的结构化程度；TAPDreamer 的有效秩在优化早期迅速下降（结构涌现），表明扰动向低维共享方向集中。

**符号说明**:
- $\sigma_j$：位移场矩阵的奇异值
- $q_j$：归一化奇异值平方（谱概率）

---

## 关键图表

### Figure 1: TAPDreamer 总览——补丁一次优化、多处复用

![Figure 1](https://arxiv.org/html/2610.06814v3/fig1b_reuse_no_boxes.png)

**说明**: (a) 离线优化阶段：仅利用公开[[Visual Encoder|视觉编码器]]和源任务帧制作补丁；(b) 复用阶段：同一固定补丁跨任务和跨 WAM 变体部署，成功破坏抽屉打开（[[FastWAM]]）、锅放置（[[Motus]]）和碗放置（[[DreamWAM]]）任务。

---

### Figure 2: 真实机器人（SO-101）实验对比

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: SO-101 机器人执行"抓香蕉放碗"任务的四种条件对比：干净、同尺寸白色补丁、TAPDreamer 位置1、TAPDreamer 位置2。TAPDreamer 使 [[FastWAM]] 从 92% 降至 0%，[[DreamWAM]] 从 80% 降至 20%。

---

### Figure 3: TAPDreamer 方法概览

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: (a) 离线优化：冻结[[Visual Encoder|视觉编码器]]，最大化 L1 潜变量位移，Adam 500 步；(b) 固定补丁部署：优化完成后补丁参数冻结，直接贴到测试环境中。

---

### Figure 4: 补丁传播机制——局部变化如何扩散到全局

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: 展示局部补丁通过 [[VAE]] 编码瓶颈的[[Attention Mechanism|注意力]]机制扩散至所有空间位置。off-footprint 变化分解为共享源模板 $\Delta\bar{Z}_\text{src}$ 和观测特定残差，解释了跨观测可迁移性。

---

### Figure 5: 空间扩散可视化与注意力恢复实验

![Figure 5](https://arxiv.org/html/2610.06814v3/fig4_colorbar_row_tnr.png)

**说明**: (a-b) 随机补丁 vs TAPDreamer 在各空间位置产生的最终潜变量变化热力图——TAPDreamer 产生全局高强度变化，随机补丁几乎局限于 footprint 内；(c) 恢复注意力权重或值矩阵到干净状态后，off-footprint 变化量对比（95% 置信区间）。

---

### Figure 6: 优化轨迹——三个随机初始化的收敛过程

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: 展示潜变量 L1 位移和源模板一致性（agreement）随优化步数的演变。虚线为随机补丁基线。三个不同初始化均收敛，表明优化稳定。

---

### Figure 7: 补丁面积消融

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: 从 3.35% 增加到 3.96% 面积时，TAPDreamer 攻击效果从 65.60% 大幅下降至 3.85%，而同面积随机补丁几乎无变化，说明**攻击效果来自优化内容而非单纯遮挡**。

---

### Figure 8: 扰动结构演化（Appendix B.2）

*（图片未在 arXiv HTML 中渲染，需从 PDF 获取）*

**说明**: 优化过程中四个维度的演变：L1 位移、模板一致性、有效秩（erank）、冻结模板 L1 下界。有效秩在早期迅速下降至低维，共享方向在早期涌现后幅值持续增长。

---

### Figure 9: 闭环行为分析——末端执行器轨迹

![Figure 9](https://arxiv.org/html/2610.06814v3/fig5c_frames_tnr.png)

**说明**: 干净执行 vs TAPDreamer 攻击下的观测序列对比（上方帧序列）和三维末端执行器轨迹对比（下方曲线）。攻击后轨迹系统性偏向下方/错误方向，与干净轨迹的分叉在早期即显现；持续暴露导致失败，移除补丁后可视任务状态部分恢复。

---

### Table 1: FastWAM 主要攻击结果

| 基准 | 干净成功率 (%) | TAPDreamer (%) | 随机补丁 (%) |
|------|--------------|----------------|------------|
| LIBERO-Spatial | 97.50 | 0.00 | ~80 |
| LIBERO-Object | 98.00 | 0.00 | ~80 |
| LIBERO-Goal | 97.50 | 0.00 | ~80 |
| LIBERO-Long | 97.60 | 0.00 | ~80 |
| **LIBERO 全部 (40任务)** | **97.65** | **0.00** | **~80** |
| **RoboTwin (50任务)** | **90.86** | **0.00** | **~80** |

**关键发现**: TAPDreamer 在 LIBERO 和 RoboTwin 上均将 [[FastWAM]] 成功率降至 0%，而匹配大小的随机补丁保持约 80%，证明攻击效果完全来自优化。

---

### Table 2: 跨模型迁移攻击结果

| 目标模型 | 干净成功率 (%) | TAPDreamer 迁移 (%) |
|---------|--------------|-------------------|
| DreamWAM-uncond (DW-U) | — | 1.45 |
| DreamWAM-joint (DW-J) | — | 1.00 |
| Motus | — | 10.60 |
| FastWAM (RoboTwin) | 90.86 | 0.00 |

**关键发现**: 同一个在 [[FastWAM]] 上优化的补丁迁移到 [[DreamWAM]] 和 [[Motus]] 后仍将成功率压至极低水平（1–10%），验证了共享[[Visual Encoder|视觉编码器]]是可迁移攻击的核心桥梁。

---

### Table 3: Off-footprint 变化量（迁移可量化指标）

| 补丁类型 | Off-footprint 变化量 |
|---------|---------------------|
| TAPDreamer | ~0.75–0.80 |
| 随机补丁 | ~0.01 |

**关键发现**: TAPDreamer 产生约 75× 倍于随机补丁的 off-footprint 潜变量变化，定量证明注意力传播的显著性。

---

### Table 4: 跨观测源模板迁移（Appendix B.2）

| 指标 | 数值 |
|------|------|
| 模板余弦相似度（held-out 观测） | 0.981 |
| 中位位移保留率 | 0.833 |

**关键发现**: 源任务推导的模板几乎完美匹配 held-out 观测，中位值保留 83.3% 的实测位移，为跨任务可迁移性提供了理论下界。

---

### Table 5: 补丁位置消融（Appendix A.2）

| 条件 | 成功率 (%) |
|------|-----------|
| 重新优化各角落位置 | 0.00（各角落均达到） |
| 移动已优化补丁到新位置 | 93.5–95.0 |

**关键发现**: 补丁是位置特定的——移动到未优化位置后攻击效果消失；但在每个固定位置重新优化后均可达到 0%，说明任意位置都可被攻击，只需针对该位置优化。

---

### Table 6: 与 VLA 补丁直接迁移对比（Appendix A.3）

| 补丁来源 | FastWAM 成功率 (%) |
|---------|------------------|
| 干净（无补丁） | 97.65 |
| UPA-RFAS（VLA 补丁迁移） | 76.55 |
| UADA（VLA 补丁迁移） | 75.85 |
| **TAPDreamer（本文）** | **0.00** |

**关键发现**: 直接迁移 VLA 补丁只能将成功率降至约 76%，而 TAPDreamer 降至 0%，证明针对共享编码器优化的专用攻击远优于通用 VLA 补丁。

---

### Table 7: 注意力恢复实验（Appendix B.1）

| 恢复操作 | 保留变化量 (%) |
|---------|--------------|
| 恢复注意力权重 $A$ 到干净状态 | ~67% 消除 |
| 恢复值矩阵 $V$ 到干净状态 | ~67% 消除 |
| 同时恢复 $A$ 和 $V$ | 更大消除 |

**关键发现**: 恢复注意力权重或值矩阵后，约 67% 的潜变量变化和动作变化被消除，直接证明注意力机制是补丁传播效应的核心载体。

---

## 实验

### 数据集 / 基准

| 基准 | 规模 | 特点 | 用途 |
|------|------|------|------|
| [[LIBERO]] | 40 任务（4 子集，各 10 任务） | 桌面操纵，Spatial/Object/Goal/Long 四类 | 主要攻击评测 |
| [[RoboTwin]] | 50 任务 | 更高复杂度的操纵任务 | 跨基准泛化验证 |
| SO-101 真实机器人 | 50 次试验/条件 | 抓香蕉放碗，单臂 | 真实世界验证 |

### 实现细节

- **公共视觉编码器**: 与 FastWAM 共享的预训练[[Visual Encoder|视觉编码器]]（具体架构未披露，通常为 DINO-V2 或 SigLIP）
- **优化器**: [[Adam优化器|Adam]]，500 步迭代
- **源帧数**: $J = 6$ 帧，单源任务
- **补丁尺寸**: LIBERO 中 81×81 像素（约占输入 6.5%），固定位置
- **目标模型**: [[FastWAM]]（主要），[[DreamWAM]]-uncond、DreamWAM-joint、[[Motus]]（迁移目标）
- **补丁行为**: 降低末端执行器位置指令、使夹爪趋向张开；持续暴露累积失败，去除后可部分恢复

---

## 批判性思考

### 优点

1. **极低查询开销**：整个攻击仅需公开编码器和 6 帧图像，无需任何目标策略交互，计算成本极低
2. **强大的跨模型迁移**：同一补丁在 4 种不同 WAM 架构上均有效，证明共享编码器是系统级漏洞
3. **机制透明**：公式化分解（Eq. 6-9）和消融实验清晰揭示了攻击生效的内在机制，不是"黑盒"攻击

### 局限性

1. **位置敏感**：补丁是位置特定的（Table 5），需要假设攻击者知道补丁位置或可以将补丁放置于特定位置
2. **Motus 迁移效果较弱**：成功率降至 10.6%（vs DreamWAM 的 1%），不同动作生成路径影响迁移效率
3. **防御路径不成熟**：论文仅提出"衰减可疑注意力键值对"作为潜在防御，但承认其有效性未经验证

### 潜在改进方向

1. **位置无关补丁**：利用[[EOT|期望变换（EOT）]]在多个位置上联合优化，提升位置鲁棒性
2. **跨编码器迁移**：探索不同视觉编码器架构之间的补丁迁移（如从 DINO 攻击 SigLIP）
3. **防御评估**：设计并评估针对注意力传播机制的专用防御（如频谱过滤、注意力正则化）

### 可复现性评估

- [ ] 代码开源
- [ ] 预训练模型
- [x] 训练细节完整（Algorithm 1 与超参数完整描述）
- [x] 数据集可获取（[[LIBERO]] 和 [[RoboTwin]] 均公开）

---

## 关联笔记

### 基于

- [[WAM]]: TAPDreamer 专门针对 WAM 类模型的共享视觉编码器攻击面
- [[FastWAM]]: 主要攻击目标，成功率从 97.65% 降至 0.00%
- [[DreamWAM]]: 迁移攻击目标（DW-U 和 DW-J 变体）
- [[对抗补丁攻击]]: 局部对抗补丁的攻击范式基础

### 对比

- [[BadWAM]]: 基于查询的 WAM 动作攻击（需要实时查询目标策略），TAPDreamer 完全无需目标策略访问
- [[UPA]]: 通用对抗补丁，直接对 VLA 迁移仅降至 76.55%，远差于 TAPDreamer 的 0%
- [[UADA]]: 另一 VLA 补丁，迁移至 WAM 效果同样有限（75.85%）

### 方法相关

- [[Visual Encoder]]: 共享视觉编码器是攻击面的核心
- [[Attention Mechanism]]: 注意力瓶颈传播机制是可迁移性的数学根源
- [[VAE]]: 视觉编码器中的 VAE 瓶颈结构
- [[EOT]]: 期望变换，用于增强补丁鲁棒性的相关技术

### 硬件/数据相关

- [[LIBERO]]: 主要评测基准（40 任务，全部降至 0%）
- [[RoboTwin]]: 跨基准验证（50 任务，降至 0%）
- [[Motus]]: 迁移目标之一（10.60% 残余成功率）

---

## 速查卡片

> [!summary] TAPDreamer (2026)
> - **核心**: 利用公开[[Visual Encoder|视觉编码器]]（零目标策略查询）优化固定局部[[对抗补丁攻击|对抗补丁]]，通过[[Attention Mechanism|注意力]]传播全局破坏 [[WAM]] 策略
> - **方法**: 最大化编码器 L1 潜变量位移（Eq. 4），Adam 500步，6帧，6.5% 面积补丁
> - **结果**: [[FastWAM]] 在 LIBERO（40任务）和 RoboTwin（50任务）均从 ~97%/91% 降至 **0%**；迁移至 DreamWAM / Motus 亦有效
> - **代码**: 未开源

---

*笔记创建时间: 2026-10-22*
