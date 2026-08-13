---
title: "FACT: Failure-Aware Causal Training for World-Action Models"
method_name: "FACT"
authors: [Quanquan Peng, Yutong Liang, Rui Yan, Nicklas Hansen, Xiaolong Wang]
year: 2026
venue: arXiv
tags: [world-action-model, failure-aware-training, video-diffusion, causal-transformer, bimanual-manipulation, flow-matching, candidate-scoring]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2608.10232
created: 2026-08-13
---

# 论文笔记：FACT: Failure-Aware Causal Training for World-Action Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | University of California San Diego |
| 日期 | August 2026 |
| 项目主页 | [fact-wam.github.io](https://fact-wam.github.io) |
| 对比基线 | [[GigaWorldPolicy0.5]], [[mu0]], [[PhiZero]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.10232) / [Code](https://github.com/Bariona/FACT) |

---

## 一句话总结

> FACT 是一个因果 [[World Action Model|世界-动作模型]]，先生成动作再条件化预测未来，通过对失败轨迹屏蔽动作模仿损失、保留世界预测监督，将失败数据纳入训练而不学习错误动作。

---

## 核心贡献

1. **因果 WAM 设计（Act-then-Imagine）**: 将动作预测置于未来视频和进度值预测之前；[[teacher forcing|教师强制]] clean action slot 使失败轨迹能监督世界分支，而不会污染策略。
2. **失败感知训练（Failure-Aware Training）**: 失败轨迹屏蔽动作模仿损失 $\mathcal{L}_a$，同时保留视频预测损失 $\mathcal{L}_I$ 和降低后的进度值目标，消除 success-biased hallucination。
3. **值引导候选打分（Value-Guided Candidate Scoring）**: 推理时生成 N 个动作候选，用条件化进度值头选出最优，提供性能-延迟的可配置权衡。

---

## 问题背景

### 要解决的问题

现有 [[World Action Model|WAM]] 在双臂操作等复杂任务中存在 **success-biased hallucination**：模型只在成功示教数据上训练，预测未来时倾向于幻想成功抓取，即使动作执行失败也如此。此外，失败轨迹数据大量被丢弃，造成数据浪费。

### 现有方法的局限

- [[GigaWorldPolicy0.5]]、[[mu0]]、[[PhiZero]] 等 WAM 均以成功演示为主要训练数据。
- 未来预测与动作预测解耦不充分：模型无法有效利用 "坏动作带来坏结果" 这一监督信号。
- 失败数据若直接加入训练，会错误地将失败动作作为模仿目标，导致策略退化。

### 本文的动机

通过 **因果顺序**（先预测动作，再以 clean action 条件化世界预测）+**掩码策略**（失败轨迹屏蔽动作损失、保留世界损失），将失败轨迹转化为有效的世界模型监督信号，同时不影响策略质量。失败数据还能为进度值头提供负样本，使其具备辨别成功与失败的能力，进而在推理时实现 [[Best-of-N|候选打分]]。

---

## 方法详解

### 模型架构

FACT 采用 **共享因果扩散 Transformer** 架构，以 [[Wan 2.2-5B|WAN2.2-5B]] 视频骨干为基础：

- **输入**: 语言指令 $\ell$ + 多视角 RGB 观测 $o_t$ + 本体感知状态
- **Backbone**: [[Wan 2.2-5B|WAN2.2-5B]] 视频扩散 Transformer
- **核心模块**: [[teacher forcing|教师强制动作槽]] 用于将 clean action 注入世界分支；[[Action Chunking|动作块]] 预测头输出 $a_{t:t+H}$
- **动作适配器**: 附加在 robot token 上的轻量 Action Adapter，保留预训练视频通路
- **输出**: 动作块 $a_{t:t+H}$、任务进度值 $v_t \in [0,1]$、未来帧 $o'_{t:t+K}$
- **参数规模**: WAN2.2-5B 量级

#### Token 序列设计

$$
z = [z^P_{\text{ref}} \;\|\; z^A_{\text{pred}} \;\|\; z^G_{\text{gt}} \;\|\; z^V_{\text{value}} \;\|\; z^I_{\text{future}}]
$$

| 槽位 | 符号 | 内容 |
|------|------|------|
| 观测前缀 | $P$ | 当前多视角 RGB + 本体感知 |
| 带噪动作 | $A$ | 扩散过程中正在去噪的动作 token |
| 干净动作 | $G$ | [[teacher forcing|教师强制]] 的 ground-truth clean action |
| 进度值 | $V$ | 任务进度预测头（条件化于 $G$） |
| 未来视频 | $I$ | 未来帧预测（条件化于 $G$） |

**关键设计**: $V$ 和 $I$ 的 attention 只能看到 $G$（clean），不看 $A$（noisy）；$A$ 不可观测 $G$。这一 [[因果混合注意力|因果注意力掩码]] 设计使失败轨迹可以监督 $V$ 和 $I$，而无需将失败动作作为模仿目标。

### 核心模块

#### 模块 1: 因果动作条件化（Causal Action Conditioning）

**设计动机**: 利用 [[Causal Transformer|因果 Transformer]] 的单向注意力，确保未来预测在动作已知后才进行，模拟真实的因果关系。

**具体实现**:
- 推理 Stage 1：对 $[P, A_{\text{noisy}}]$ 执行 $K_{\text{denoise}}$ 步 flow-Euler 去噪，得到 $\hat{a}$
- 推理 Stage 2：将 $\hat{a}$ 放入 clean slot $G$，对 $V$ 和 $I$ 进行去噪
- [[teacher forcing|教师强制]] 在训练时直接用 ground-truth action 填充 $G$

#### 模块 2: 失败感知训练（Failure-Aware Training）

**设计动机**: 失败轨迹提供了"坏动作 → 坏结果"的因果监督信号，但不应将失败动作纳入模仿目标。

**具体实现**:
- 成功轨迹：正常监督所有损失（动作 + 进度 + 未来）
- 失败轨迹：**屏蔽动作模仿损失** $\mathcal{L}_a$，保留 $\mathcal{L}_v$（目标下调）和 $\mathcal{L}_I$
- 进度值目标对失败轨迹施加惩罚：$v_t = \text{clip}(p_{t+H} - \lambda_{\text{fail}} \cdot \mathbf{1}_{\text{fail}}, 0, 1)$

#### 模块 3: 值引导候选打分（Value-Guided Candidate Scoring）

**设计动机**: 训练好的进度值头已具备辨别好坏动作的能力，可在推理时作为打分函数。

**具体实现**:
- 采样 $N$ 个动作候选（独立并行 Stage 1 去噪）
- 用 Stage 2 的值头为每个候选打分，选取最高分
- $N=4$ 时性能-延迟权衡最优

---

## 关键公式

### 公式 1: [[World Action Model|WAM 问题形式化]]

$$
p_\theta(o'_{t:t+K},\, v_t(a_{t:t+H}) \mid o_t, \ell, a_{t:t+H})
$$

**含义**: 给定当前观测 $o_t$、指令 $\ell$ 和执行动作 $a_{t:t+H}$，预测未来帧序列 $o'$ 和任务进度值 $v$。

**符号说明**:
- $o_t$: 当前多视角观测（RGB + 本体感知）
- $\ell$: 语言任务指令
- $a_{t:t+H}$: 执行的动作块（chunk length $H=48$）
- $o'_{t:t+K}$: 未来 $K$ 帧预测
- $v_t \in [0,1]$: 归一化任务进度值

### 公式 2: [[Flow Matching|Flow Matching 去噪损失]]

$$
\mathcal{L}_x = \mathbb{E}_{z^x_0,\, z^x_1,\, \tau}\!\left[\left\|u^x_\theta(z^x_\tau, \tau;\, z) - (z^x_1 - z^x_0)\right\|^2_2\right]
$$

**含义**: 对每个模态 $x \in \{a, v, I\}$ 独立施加 flow-matching 去噪损失，通过预测 velocity field 学习从噪声分布到数据分布的确定性轨迹。

**符号说明**:
- $z^x_0$: 噪声样本（标准正态）
- $z^x_1$: 干净数据样本（latent）
- $\tau \sim \mathcal{U}(0,1)$: 插值时刻
- $z^x_\tau = (1-\tau)z^x_0 + \tau z^x_1$: 插值中间状态
- $u^x_\theta$: 网络预测的速度向量

### 公式 3: [[teacher forcing|成功轨迹训练损失]]

$$
\mathcal{L}_{D_s} = w_a \cdot \mathcal{L}_a + w_v \cdot \mathcal{L}_v + w_I \cdot \mathcal{L}_I
$$

**含义**: 成功演示同时监督动作、进度值和未来视频三个目标，动作损失权重远高于其余两项。

**符号说明**:
- $w_a = 20$: 动作损失权重（主导项）
- $w_v = 1$: 进度值损失权重
- $w_I = 1$: 未来视频损失权重

### 公式 4: 失败轨迹训练损失（Failure-Aware 掩码）

$$
\mathcal{L}_{D_f} = w_v \cdot \mathcal{L}_v + w_I \cdot \mathcal{L}_I
$$

**含义**: 失败轨迹屏蔽动作模仿损失 $\mathcal{L}_a$，仅保留世界预测损失，防止策略学习错误动作。

### 公式 5: 失败感知进度值目标

$$
v_t(a^{\text{gt}}_{t:t+H}) = \begin{cases} p_{t+H} & \text{if success} \\ \text{clip}(p_{t+H} - \lambda_{\text{fail}} \cdot \mathbf{1}_{\text{fail}}(t+H),\; 0, 1) & \text{if fail} \end{cases}
$$

**含义**: 对失败时刻的进度值施加惩罚偏移，使值头学会区分成功与失败状态。

**符号说明**:
- $p_{t+H} = G_{t+H}/G_T$: 基于累积收益的归一化进度
- $\lambda_{\text{fail}} = 1$: 失败惩罚强度（设为 1，将进度值拉低至 0 附近）
- $\mathbf{1}_{\text{fail}}(t+H)$: 该时刻是否为失败状态的指示函数

### 公式 6: [[Best-of-N|值引导候选打分]]

$$
a^* = \arg\max_{k \in \{1,\ldots,N\}} \hat{v}^{(k)} = \arg\max_{a \in a^{(1:N)}} V_\theta(o_t, \ell, a)
$$

**含义**: 在 $N$ 个并行采样的动作候选中，选取进度值预测最高者作为最终执行动作。

**符号说明**:
- $N$: 候选动作数量（实验最优 $N=4$）
- $\hat{v}^{(k)}$: 第 $k$ 个候选的预测进度值
- $V_\theta$: 条件化进度值函数（以 $o_t, \ell, a$ 为输入）

---

## 关键图表

### Figure 1: 整体架构概览（Act-then-Imagine）

![FACT Architecture](https://fact-wam.github.io/static/images/method/frame_14.png)

**说明**: FACT 的整体框架。共享因果扩散 Transformer 依次去噪动作 $A$、进度值 $V$、未来帧 $I$；$V$ 和 $I$ 通过 [[Causal Transformer|因果注意力掩码]] 条件化于 clean action slot $G$，而非带噪的 $A$。两阶段推理：Stage 1 完成动作去噪，Stage 2 以干净动作为条件预测世界状态。

### Figure 2: Token 注意力掩码（训练 vs 推理）

> arXiv HTML 暂时无法访问（论文刚提交，服务器返回 503），暂缺该图外链。

**说明**: 展示训练时 $[P, A, G, V, I]$ 各 token 槽之间的 attention 可见性。$A$ 看不到 $G$；$V$ 和 $I$ 只能看到 $G$（不看 $A$）。推理时无 $G$，Stage 1 去噪 $A$ 后填入 $G$ 槽再执行 Stage 2。

### Figure 5: 失败预测对比（Success-Only vs FACT）

![Failure Prediction Comparison](https://fact-wam.github.io/static/images/future_pred.webp)

**说明**: 在相同坏动作输入下，success-only 模型幻想出成功抓取的未来帧（虚线框）；而 FACT 正确预测了实际发生的失败状态（物体未被抓起）。failure-aware 训练消除了 success-biased hallucination，PSNR 提升 +6.4 dB。

### Figure 6: 失败数据量化扩展曲线

> arXiv HTML 暂时无法访问，暂缺该图外链。

**说明**: 在 RoboTwin 三个任务上，随失败数据比例 $p$ 从 0% 增至 100%，平均成功率从 32.7% 单调上升至 57.3%。说明 FACT 框架下失败数据量越大收益越高。

### Figure 8: 进度值轨迹示例

> arXiv HTML 暂时无法访问，暂缺该图外链。

**说明**: 可视化执行一段双臂任务时预测进度值随时间的变化。值在成功操作时稳步上升，在抓取失败时骤降，在恢复动作后回升，验证了值头对动作结果的判别能力。

---

### Table 1: RoboTwin 仿真测试（50 任务，平均成功率 %）

| Method | Clean | Rand. | Average |
|--------|-------|-------|---------|
| π₀ | 65.9 | 58.4 | 62.2 |
| X-VLA | 72.9 | 72.8 | 72.9 |
| π₀.₅ | 82.7 | 76.8 | 79.8 |
| Gigaworld-Policy | 87.0 | 85.0 | 86.0 |
| Motus | **88.7** | **87.0** | **87.8** |
| FACT（无 failure） | 86.3 | 84.9 | 85.6 |
| **FACT w/ failure** | 88.4 | 86.6 | **87.5** |

**关键发现**: 视频联合训练使基线从 81.8% 提升到 85.6%；加入失败数据后达 87.5%，与 Motus 性能相当但推理速度快约 3×。

### Table 2: 真实世界双臂操作（成功率 %，共 20 次试验/任务）

| Method | Stack Cubes | Pick Cubes | Hand over | Stack Bowls | Pour | Avg. |
|--------|-------------|-----------|-----------|-------------|------|------|
| Cosmos | 5 | 45 | 25 | 35 | 15 | 25 |
| π₀ | 35 | 70 | 40 | 50 | 45 | 48 |
| π₀.₅ | 75 | 100 | 85 | 80 | 100 | **88** |
| Motus | 50 | 70 | 55 | 85 | 60 | 64 |
| FACT | 70 | 85 | 90 | 80 | 85 | 82 |
| FACT w/ failure | 75 | 95 | 85 | 95 | 95 | 89 |
| **FACT w/ failure + scoring** | **85** | **100** | 85 | **100** | **90** | **92** |

**关键发现**: 失败数据将实测成功率从 82% → 89%；叠加 Best-of-N 打分达 92%，超越 π₀.₅。

### Table 3: 泛化测试（未见任务变体，成功率 %）

| Method | Stack Cubes | Pick Cubes | Stack Bowls | Avg. |
|--------|-------------|-----------|-------------|------|
| π₀ | 30 | 65 | 75 | 57 |
| π₀.₅ | 65 | 90 | 100 | 85 |
| Motus | 55 | 60 | 70 | 62 |
| FACT | 45 | 75 | 80 | 67 |
| FACT w/ failure | 60 | 85 | 85 | 77 |
| **FACT w/ failure + scoring** | **65** | **95** | 85 | **82** |

**关键发现**: 在未见任务变体上，failure data 将泛化能力从 67% 提升至 82%，说明失败监督有助于习得更鲁棒的策略。

### Table 4: 失败预测质量消融（PSNR，dB）

| 子集 | FACT（无 failure） | FACT w/ failure |
|------|------------------|-----------------|
| 全部样本 | 22.82 | **26.00** |
| 成功轨迹 | 26.12 | 26.08 |
| 失败轨迹 | 19.51 | **25.92** |

**关键发现**: 失败感知训练大幅提升对失败轨迹的预测质量（+6.4 dB），成功轨迹预测几乎不受影响，说明两类数据的监督信号是互补而非冲突的。

### Table 5: 消融实验（真实世界 seen tasks 平均成功率 %）

| 配置 | Avg. | 说明 |
|------|------|------|
| w/o video co-training | 58% | 视频联合训练至关重要 |
| w/o causal mask | 77% | 因果掩码对失败数据利用不可或缺 |
| failure data w/ action loss | 63% | 直接模仿失败动作会损害策略 |
| **Full FACT** | **82%** | 所有组件缺一不可 |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[RoboTwin]] | 50 任务 | 仿真双臂，含 clean/domain-randomized 两种 split | 仿真训练 & 测试 |
| Real-world bimanual | 5 seen + 3 unseen 变体 | 实验室真实双臂操作环境 | 真实测试 |
| Failure rollout data | ~1300（仿真）/ ~30/任务（真实） | 从模型自身rollout获取的失败轨迹 | 失败感知训练 |

### 实现细节

- **Backbone**: [[Wan 2.2-5B|WAN2.2-5B]] 视频扩散 Transformer（5B 参数级别）
- **优化器**: AdamW；Action FFN 学习率 $2\times10^{-4}$，WAN backbone $2\times10^{-5}$
- **动作块长度**: $H = 48$
- **推理步数**: 20 步 flow-Euler（Stage 1 + Stage 2 各约 10 步）
- **未来帧监督时刻**: $[0, H/4, H/2, 3H/4, H]$（5 帧稀疏监督）
- **候选打分**: $N=4$（最优性能-延迟权衡）
- **失败数据来源**: 主要来自模型自身部署时的失败 rollout（主动数据收集，~30次/真实任务）

---

## 批判性思考

### 优点

1. **设计简洁优雅**: 因果顺序（act-then-imagine）在架构层面自然解决了失败数据的利用问题，无需额外的判别器或数据筛选。
2. **真实数据效率高**: 仅需约 30 次/任务的失败 rollout 即可显著提升真实世界表现，成本极低。
3. **模块化部署灵活**: scoring 是可选组件，在延迟敏感场景可直接关闭，基础策略仍有竞争力。
4. **消除 hallucination 的副产品**: 改善未来预测质量（failure PSNR +6.4 dB），对下游规划器或监控系统也有价值。

### 局限性

1. **仅在双臂操作任务测试**: 未验证在自主导航、足式运动等其他机器人任务上的泛化性。
2. **失败数据仍需主动收集**: 虽然量少，但收集失败 rollout 在真实机器人上有一定风险和人力成本，自动化程度有限。
3. **泛化能力仍逊于 π₀.₅**: 在未见任务变体上，FACT+scoring（82%）仍低于 π₀.₅（85%），说明预训练数据规模仍是关键因素。
4. **视频生成开销**: WAN2.2-5B 基础推理成本高，两阶段推理 + Best-of-N 在低计算资源场景仍是挑战。

### 潜在改进方向

1. 将失败数据来源扩展到人类演示的失败片段或人机交互过程，减少主动收集成本。
2. 将值头替换为更精细的奖励模型（如 VLM 辅助评分），提升 Best-of-N 的选择精度。
3. 探索在更大规模互联网视频数据上预训练世界分支，增强对物理后果的感知能力。

### 可复现性评估

- [x] 代码开源（[github.com/Bariona/FACT](https://github.com/Bariona/FACT)）
- [ ] 预训练模型权重（未确认）
- [x] 训练细节完整（优化器、超参数均在论文中列出）
- [ ] 数据集完整公开（失败 rollout 数据未公开）

---

## 关联笔记

### 基于

- [[GigaWorldPolicy0.5]]: 同类 WAM 基线，FACT 在仿真上超越之
- [[mu0]] ([[PhiZero]]): 真实世界对比基线（π₀、π₀.₅）
- [[Wan 2.2-5B]]: 视频扩散 Transformer 骨干网络
- [[RoboTwin]]: 主要仿真评测平台（50 任务）

### 对比

- [[Kairos]]: 另一个关注 regret/失败感知的 WAM，关注点更侧重规划
- [[Flash-WAM]]: 关注推理速度的 WAM，FACT 声称 ~3× 速度优势相对于 Motus 类
- [[RepWAM]]: 基于 tokenizer 的 WAM 方案

### 方法相关

- [[World Action Model|WAM]]: 核心框架
- [[Flow Matching]]: 所有模态的去噪目标
- [[teacher forcing|Teacher Forcing]]: 因果动作条件化的关键机制
- [[Action Chunking]]: 动作预测的基本形式（chunk length H=48）
- [[Best-of-N]]: 值引导候选打分的推理策略
- [[Causal Transformer]]: 注意力掩码设计的基础

### 数据集相关

- [[RoboTwin]]: 仿真评测（50 任务，clean + domain-randomized）

---

## 速查卡片

> [!summary] FACT: Failure-Aware Causal Training for World-Action Models
> - **核心**: 因果 WAM，先 act 再 imagine，失败数据只督导世界预测，不模仿失败动作
> - **方法**: 共享视频 DiT + 教师强制 clean action slot + 失败轨迹掩码动作损失
> - **结果**: RoboTwin 87.5% / 真实世界 89%（92% w/ scoring），推理速度 3× Motus
> - **代码**: [github.com/Bariona/FACT](https://github.com/Bariona/FACT)

---

*笔记创建时间: 2026-08-13*
