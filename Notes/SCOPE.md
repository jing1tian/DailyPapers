---
title: "SCOPE: Score-Isolated Agentic Optimization for Video World Models"
method_name: "SCOPE"
authors: [Yuhua Jiang, Jiaming Wang, Qingbin Liu, Feifei Gao]
year: 2026
venue: arXiv
tags: [video-world-model, inference-time-adaptation, score-isolation, video-generation, agentic-optimization, diffusion-model, physics-reasoning]
zotero_collection: ""
image_source: online
arxiv_html: https://arxiv.org/html/2608.15043
created: 2026-08-19
---

# 论文笔记：SCOPE: Score-Isolated Agentic Optimization for Video World Models

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Tsinghua University, National University of Singapore, Tencent |
| 日期 | August 2026 |
| 项目主页 | — |
| 对比基线 | [[Wan2.2]], [[CogVideoX]] (Frozen Base) |
| 链接 | [arXiv](https://arxiv.org/abs/2608.15043) / [Code](https://github.com/YuhuaJiang2002/SCOPE) |

---

## 一句话总结

> SCOPE 提出了一种可审计的推理时自适应框架，通过**分数隔离**原则将冻结的[[Video World Model|视频世界模型]]的控制状态更新与持出测试集分数严格隔离，在 [[Physics-IQ Verified|Physics-IQ]] 基准上提升高达 +14.24 分。

---

## 核心贡献

1. **推理-控制-评估分离**: 明确区分"提议的更新"、"已部署的控制状态"和"持出评估分数"三者，解决传统方法将三者混淆导致的数据泄露问题
2. **类型化控制状态 + 四轴分解**: 将控制状态分解为文本指令、采样配置、验证/弃权规则、奖励选择机制四个正交维度，每维度独立提议和更新
3. **可溯源审计链**: 通过密码学哈希对每次状态转移进行绑定记录，保证任何状态的演化历史完全可验证

---

## 问题背景

### 要解决的问题

当前对冻结[[Video World Model|视频世界模型]]进行[[Inference-Time Adaptation|推理时自适应]]时，存在**推理-控制-评估间隙 (inference-control evaluation gap)**：系统的多个组件同时变化，无法归因性能提升来源，且测试集反馈可能渗入已部署系统。

### 现有方法的局限

- **Qwen-Image-Agent 风格反馈**方法：允许测试集评估分数影响控制状态更新，违背了公正评估的原则
- **LingBot 风格 Director→Pilot** 方法：将整体策略与具体生成混合，难以追踪每次变更的具体贡献
- **VIGOR reward-only** 等纯奖励方法：无验证/弃权机制，无法排除有害更新
- 所有现有方法均缺乏**形式化的分数隔离**保证和可溯源审计链

### 本文的动机

通过将控制状态的开发时更新限制在开发集上，并在评估前冻结路由函数，可以保证持出评估结果对控制状态、路由决策、更新接受标准均只读，从而在提升性能的同时维持评估完整性。

---

## 方法详解

### 模型架构

SCOPE 采用**类型化状态机**架构，外部控制以类型化状态表示，通过有界变更机制更新，评估前冻结策略：

- **输入**: 视频生成场景 $s$，初始控制状态 $\Omega_0$，开发任务集 $D_r$
- **Backbone**: 任意冻结[[Video World Model|视频世界模型]]（实验中为 [[Wan2.2]] 和 [[CogVideoX]]）
- **核心模块**: [[Score Isolation|分数隔离]] + [[Inference-Time Adaptation|推理时状态更新]] + [[Provenance Binding|可溯源绑定]]
- **输出**: 冻结路由函数 $\Phi_{\Omega_R}$，对每个场景选择最优候选视频

### 核心模块

#### 模块1: 类型化控制状态（Typed Control State）

**设计动机**: 利用[[类型化状态]]将不同性质的控制解耦，避免相互干扰，支持独立提议和独立验证。

**具体实现**:
- $\Omega_r^{\text{text}}$: 文本指令（如场景描述物理规律的 SceneLang 卡片）
- $\Omega_r^{\text{sample}}$: 采样配置（如噪声调度方案，用不同随机种子扩展候选集）
- $\Omega_r^{\text{verify}}$: 验证与弃权规则（格式检查、[[SAM]] 分割质量阈值等）
- $\Omega_r^{\text{select}}$: 奖励选择机制（[[Learned Reward Selector|学习型奖励选择器]]，基于 OOF 奖励）

#### 模块2: 提交-或-保留更新规则（Commit-or-Retain）

**设计动机**: 确保每轮更新仅在开发集目标严格提升且所有守护条件通过时才接受，从而防止有害更新滑入部署。

**具体实现**:
- 计算开发集增益 $\Delta_r^{\text{dev}}$
- 评估所有 $K$ 个守护约束 $G_{r,k}$（联合取积）
- 满足条件则接受新状态，否则精确保留旧状态（回退到 Frozen Base 的 fallback 保障也在此处体现）

#### 模块3: 冻结部署与分数隔离

**设计动机**: 经过 $R$ 轮开发后锁定控制状态，保证持出评估分数对已部署系统只读，消除测试集泄露。

**具体实现**:
- 预承诺完整路由函数 $\Phi_{\Omega_R}$，包含精确的 Frozen Base 候选 $a_s^{\text{Base}}$
- 持出评估不影响最终控制状态、路由决策和更新接受标准
- 每次转移通过密码学哈希记录，形成可审计的[[Provenance Binding|溯源绑定]]链

#### 模块4: 工具增强提议（Tool-Augmented Proposal）

**设计动机**: 利用外部知识提升文本轴的提议质量，通过检索和[[Grounded-SAM|分割工具]]辅助视觉轴的验证。

**具体实现**:
- **检索轴**: 网页检索物理概念卡（SceneLang）并作为文本条件注入，无参考像素或视觉嵌入介入生成
- **SAM 分割轴**: 使用[[Grounded-SAM]] 对生成视频进行对象分割，辅助验证物体动态轨迹

---

## 关键公式

### 公式1: [[类型化控制状态|控制状态定义]]

$$
\Omega \in \mathcal{C}
$$

**含义**: 控制状态 $\Omega$ 属于类型化控制空间 $\mathcal{C}$，是外部可调整参数的形式化表示。

**符号说明**:
- $\Omega$: 控制状态，包含文本、采样、验证、选择四个分量
- $\mathcal{C}$: 合法控制状态的类型化空间

### 公式2: [[Inference-Time Adaptation|开发时更新算子]]

$$
\Omega_{r+1} = \mathcal{U}_r(\Omega_r, \delta_r; D_r), \quad r = 0, \ldots, R-1
$$

**含义**: 第 $r$ 轮更新算子 $\mathcal{U}_r$ 基于当前状态 $\Omega_r$、提议变更 $\delta_r$ 和开发集 $D_r$ 产生新状态。

**符号说明**:
- $\mathcal{U}_r$: 第 $r$ 轮的更新算子（实现为提交-或-保留规则）
- $\delta_r$: 第 $r$ 轮提议的变更（轴特定提议）
- $D_r$: 第 $r$ 轮使用的开发任务集
- $R$: 总开发轮数

### 公式3: [[Score Isolation|冻结部署策略]]

$$
\pi_{\Omega_R}(s) = F(\Omega_R, x_s, \mathcal{A}_s)
$$

**含义**: 经过 $R$ 轮开发后，控制状态锁定为 $\Omega_R$，路由函数 $F$ 从候选集 $\mathcal{A}_s$ 中选择最优输出。

**符号说明**:
- $\Omega_R$: 冻结后的最终控制状态
- $x_s$: 场景 $s$ 的上下文信息
- $\mathcal{A}_s$: 候选视频集合（含 Frozen Base 候选）
- $F$: 奖励选择路由函数

### 公式4: [[类型化控制状态|控制状态四轴分解]]

$$
\Omega_r = (\Omega_r^{\text{text}}, \Omega_r^{\text{sample}}, \Omega_r^{\text{verify}}, \Omega_r^{\text{select}})
$$

**含义**: 控制状态分解为文本、采样、验证、选择四个正交轴，每轴独立演化。

**符号说明**:
- $\Omega_r^{\text{text}}$: 文本指令轴（SceneLang 物理卡片等）
- $\Omega_r^{\text{sample}}$: 采样配置轴（噪声调度、随机种子策略）
- $\Omega_r^{\text{verify}}$: 验证/弃权轴（格式与质量守护规则）
- $\Omega_r^{\text{select}}$: 奖励选择轴（候选视频排序策略）

### 公式5: 轴特定提议

$$
\delta_r = P_r(\Omega_r, D_r)
$$

**含义**: 提议机制 $P_r$（可替换的"灰带"模块）基于当前状态和开发集生成变更提议 $\delta_r$。

**符号说明**:
- $P_r$: 提议机制（如 GPT-5.4 生成 SceneLang 卡片、噪声调度搜索等）
- $\delta_r$: 针对某一轴的有界变更提议

### 公式6: [[开发集目标|开发对比增益]]

$$
\Delta_r^{\text{dev}}(D_r) = J_{\text{dev}}(U_r(\Omega_r, \delta_r); D_r) - J_{\text{dev}}(\Omega_r; D_r)
$$

**含义**: 衡量接受提议 $\delta_r$ 后在开发集上的性能增益，正值时才考虑接受更新。

**符号说明**:
- $J_{\text{dev}}(\cdot; D_r)$: 开发集目标函数（如平均 Physics-IQ 得分）
- $U_r(\Omega_r, \delta_r)$: 将提议应用到当前状态后的候选新状态

### 公式7: 可容许性约束（守护条件）

$$
G_{r,k}(U_r(\Omega_r, \delta_r); D_r) \leq 0, \quad k = 1, \ldots, K
$$

**含义**: 每个守护条件 $G_{r,k}$ 必须满足（结果 $\leq 0$），用于防止格式违规、质量退化等有害更新。

**符号说明**:
- $G_{r,k}$: 第 $k$ 个守护约束函数
- $K$: 守护条件总数

### 公式8: 守护条件合取

$$
G_r(D_r) = \prod_{k=1}^K \mathbb{1}[G_{r,k}(U_r(\Omega_r, \delta_r); D_r) \leq 0]
$$

**含义**: 所有守护条件取积（AND），仅当全部满足时 $G_r = 1$，更新才可能被接受。

**符号说明**:
- $\mathbb{1}[\cdot]$: 示性函数
- $G_r(D_r) = 1$: 所有守护条件均通过

### 公式9: [[Score Isolation|提交-或-保留规则]]

$$
\Omega_{r+1} = \begin{cases} U_r(\Omega_r, \delta_r), & \Delta_r^{\text{dev}}(D_r) > 0 \land G_r(D_r) = 1 \\ \Omega_r, & \text{otherwise} \end{cases}
$$

**含义**: SCOPE 的核心更新机制——仅在开发集增益严格为正且全部守护条件通过时才接受提议，否则精确保留当前状态（包含 Frozen Base fallback）。

**符号说明**:
- $\Delta_r^{\text{dev}}(D_r) > 0$: 开发集上有严格正向增益
- $G_r(D_r) = 1$: 所有守护条件通过
- $U_r(\Omega_r, \delta_r)$: 候选新状态

### 公式10: 可容许候选集

$$
\mathcal{A}_s = \{a_s^{(1)}, \ldots, a_s^{(M)}, a_s^{\text{Base}}\}
$$

**含义**: 对场景 $s$，候选集包含 $M$ 个增强候选和精确的 Frozen Base 候选，保证最差情况回退。

**符号说明**:
- $a_s^{(i)}$: 第 $i$ 个增强生成候选（来自不同噪声/文本配置）
- $a_s^{\text{Base}}$: 精确的 Frozen Base 输出（保底回退）

### 公式11: 固定部署路由

$$
a_s^{\star} = \pi_{\Omega_R}(s) = F(\Omega_R, x_s, \mathcal{A}_s)
$$

**含义**: 在冻结控制状态 $\Omega_R$ 下，路由函数对每个场景选择最优候选输出。

**符号说明**:
- $a_s^{\star}$: 最终选择的视频输出
- $\pi_{\Omega_R}$: 冻结后的部署策略

### 公式12: 预承诺部署映射

$$
\Phi_{\Omega_R}: (s, \mathcal{A}_s) \longmapsto a_s^{\star}
$$

**含义**: 完整路由函数在评估前被预先承诺，持出评估分数无法回流影响映射。

**符号说明**:
- $\Phi_{\Omega_R}$: 预承诺的确定性路由映射

### 公式13: [[Provenance Binding|溯源绑定记录]]

$$
\mathcal{T}_r = (h(\Omega_r),\; h(\delta_r),\; h(D_r),\; b_r,\; \mathcal{R}_r,\; h(\Omega_{r+1}))
$$

**含义**: 第 $r$ 轮的完整审计记录，通过密码学哈希将父状态、提议、开发数据、提交决定和后继状态绑定。

**符号说明**:
- $h(\cdot)$: 密码学哈希函数
- $b_r \in \{0, 1\}$: 提交决定（1=接受，0=拒绝）
- $\mathcal{R}_r$: 该轮守护条件结果记录
- $h(\Omega_{r+1})$: 后继状态哈希

### 公式14: 可容许父状态检查

$$
h(\Omega_r^{\text{parent}}) = h(\Omega_r)
$$

**含义**: 审计时验证父状态哈希一致，确保链完整性，防止中间状态被篡改。

**符号说明**:
- $h(\Omega_r^{\text{parent}})$: 记录的父状态哈希
- $h(\Omega_r)$: 当前状态实际哈希

### 公式15: 拒绝提议蕴含关系

$$
b_r = 0 \implies \Omega_{r+1} = \Omega_r \implies h(\Omega_{r+1}) = h(\Omega_r)
$$

**含义**: 当提议被拒绝时，后继状态与当前状态完全相同，哈希也相同，审计链可验证。

**符号说明**:
- $b_r = 0$: 本轮提议被拒绝
- $\Omega_{r+1} = \Omega_r$: 状态未发生任何变化

---

## 关键图表

### Figure 1: Overview / SCOPE 系统概览

*（arXiv HTML 中 Figure 1 为复杂示意图，暂无直接外链 URL。见论文原图。）*

**说明**: SCOPE 整体框架概览。灰带中的提议机制可替换；框架维护四轴类型化控制状态，提供分数盲部署选择（含精确 Base 回退）、开发时有界状态更新和分数隔离约束（持出结果只读）。

### Figure 2: Policy Evolution Across Backbones / 跨 backbone 的策略演化

![Figure 2: Policy Evolution](https://arxiv.org/html/2608.15043v1/figures/scope_common_base_first_four_rounds.png)

**说明**: 两个冻结 backbone（[[Wan2.2]] 和 [[CogVideoX]]）上的 Physics-IQ 策略演化。R0–R3 分别对应 Base、物理卡片更新、规则精化更新、最终 SCOPE 状态。曲线为 40 个场景的平均 Physics-IQ 得分，阴影为 95% 场景自举置信区间。关键观察：每轮更新均有单调提升，且两个 backbone 的演化趋势高度一致，说明文本轴更新具有跨模型迁移性。

### Figure 3: Retrieval-Assisted Generation Example / 检索辅助生成示例（Newton's Cradle）

![Figure 3: Retrieval Example](https://arxiv.org/html/2608.15043v1/figures/newton_web_case_comparison.png)

**说明**: 牛顿摆场景的检索辅助生成示例。检索提供文本概念卡（SceneLang），无参考像素或视觉嵌入介入生成过程。展示了文本轴提议如何通过物理知识注入改善视频中的力学行为表现。

### Figure 4: Web-Grounded Post-Hoc Intervention / 网络锚定的事后干预（完整纸张浸水）

![Figure 4: SAM Intervention](https://arxiv.org/html/2608.15043v1/figures/sam_web_case_frames.png)

**说明**: 网络锚定的完整纸张浸水干预可视化。两行均使用相同的无吸管种子基线。该可视化为定性的事后状态干预，并非原生 [[Wan2.2]] 动态。展示了 [[Grounded-SAM]] 分割辅助的验证轴如何介入筛选生成视频。

### Table 1: Common-Base Physics-IQ比较（主要结果）

| 方法 | Wan P-IQ | Wan Δ (95% CI) | CogVideoX P-IQ | CogVideoX Δ (95% CI) |
|------|----------|----------------|----------------|----------------------|
| Frozen Base | 20.70 | — | 22.91 | — |
| WMReward reward-only | 21.81 | +1.11 [-0.40, +3.17] | 26.10 | +3.19 [+0.85, +6.04] |
| VAE-θ reward | 22.33 | +1.63 [-0.67, +4.46] | 25.14 | +2.23 [+0.02, +4.83] |
| VIGOR reward-only | 25.15 | +4.44 [+1.49, +7.96] | 27.75 | +4.83 [+1.71, +8.44] |
| Uniform random 3-candidate selector | 27.41 | +6.71 [+1.93, +11.45] | 30.57 | +7.65 [+4.51, +10.97] |
| LingBot-style Director→Pilot proxy | 29.98 | +9.27 [+2.37, +16.12] | 34.91 | +12.00 [+6.76, +17.62] |
| Fixed GPT-5.4 physics card | 30.04 | +9.33 [+2.50, +15.93] | 34.89 | +11.98 [+7.05, +17.13] |
| Fixed rule refinement | 31.50 | +10.80 [+2.67, +19.14] | 33.89 | +10.98 [+6.27, +15.93] |
| Joint self-score ablation | 32.07 | +11.37 [+3.38, +19.69] | 34.63 | +11.71 [+6.83, +16.66] |
| Qwen-Image-Agent-style feedback | 32.87 | +12.16 [+4.48, +20.06] | 34.99 | +12.08 [+7.36, +16.98] |
| **SCOPE (ours)** | **34.94** | **+14.24 [+8.10, +21.23]** | **35.51** | **+12.60 [+7.73, +17.60]** |

**表格说明**: SCOPE 在 [[Physics-IQ Verified|Physics-IQ]] 基准（40 场景，common-base 设置）上取得最高得分。注意 Qwen-Image-Agent 风格方法分数（32.87）高但违反了分数隔离——该方法允许测试集分数回流影响状态，不可信。SCOPE 在保持评估完整性的前提下仍取得最优成绩。

### Table 2: Cross-Backbone 结果 —— P-AI (PAI-Bench-G)

| 匹配方法 | Wan Overall | Wan Δ (95% CI) | CogVideoX Overall | CogVideoX Δ (95% CI) |
|----------|------------|----------------|-------------------|----------------------|
| Frozen Base | 76.67 | — | 74.10 | — |
| + shared global | 76.86 | +0.19 [-2.13, +2.66] | 74.41 | +0.30 [-2.15, +2.77] |
| LingBot Director→Pilot + shared global | 76.77 | +0.10 [-2.25, +2.54] | 74.29 | +0.18 [-2.30, +2.69] |
| Qwen-Image-Agent-style (SceneLang) | 77.02 | +0.35 [-1.50, +2.32] | **74.72** | **+0.62 [-1.37, +2.68]** |
| SceneLang + shared global | 77.01 | +0.34 [-1.86, +2.63] | 71.57 | -2.53 [-5.25, +0.12] |
| SCOPE post-hoc upper envelope | **77.47** | **+0.80 [-1.38, +3.12]** | 74.48 | +0.38 [-1.56, +2.37] |

**表格说明**: [[PAI-Bench]] 上的跨 backbone 结果。所有增益的置信区间均包含零，统计上未解决。shared-global 变体在两个 backbone 间方向相反（Wan +0.19 vs CogVideoX -2.53），表明效果具有强烈的 backbone 依赖性。

### Table 3: Cross-Backbone 结果 —— OpenS2V-Eval (n=180/backbone)

| 方法 | Wan Mean | Wan Weighted | CogVideoX Mean | CogVideoX Weighted |
|------|----------|--------------|----------------|--------------------|
| Frozen Base | 37.46 | 36.60 | 48.30 | 48.29 |
| Qwen-Image-Agent-style | 38.72 | 37.92 | 46.73 | 46.16 |
| LingBot Director→Pilot prompt proxy | 41.27 | 42.84 | 47.39 | 49.62 |
| GPT-5.4 first-frame skill | 38.31 | 39.33 | 46.67 | 47.99 |
| **SCOPE (ours)** | **41.75** | **43.17** | **47.76** | 49.22 |

**表格说明**: OpenS2V-Eval 上 SCOPE 在 Wan backbone 上取得最高均值（41.75）和加权分（43.17），但 CogVideoX 上的改善仍在统计上未解决，提示 backbone-metric 交互效应。

### Table 4: 前瞻性评估结果（Prospective Evaluations）

| 评估 | 参照/标准 | 结果 |
|------|-----------|------|
| 新鲜 Physics-IQ | Base | +0.096 [-0.887, +1.077]（未解决） |
| 新鲜 Physics-IQ | 冻结路由器 | -1.019 [-2.159, +0.146]（未解决） |
| 类型化更新 | 随机更新 | -0.316 [-5.984, +5.145]（未解决） |
| 工具路由器 | 始终启用 | +0.660 [+0.016, +1.315]（已解决） |
| 工具路由器 | 无工具 | +0.083 [-0.191, +0.381]（未解决） |
| 工具路由器 | 随机路由 | +0.002 [-0.281, +0.299]（未解决） |
| PhyGround | 最低提议覆盖率: 8/32 | 4/32（低于阈值） |

**表格说明**: 前瞻性评估揭示了 SCOPE 的主要局限。在新鲜 Physics-IQ 任务上提升统计上未解决（+0.096），场景退化率（15%）超过 10% 阈值。PhyGround 中仅 4/32 任务产生有效提议（最低要求 8/32），提议生成阶段失败。

### Table 5: SCOPE 消融实验（Ablation Studies）

| 类别 | 变体 | 对比 | Δ (95% CI) |
|------|------|------|------------|
| 控制 | 文本 (SceneLang) | vs. Base | +5.62 [+0.55, +10.55] |
| 控制 | 噪声采样器 | vs. 固定采样器 | +1.26 [+0.52, +2.01] |
| 控制 | 奖励选择器 | vs. 匹配随机 | +3.33 [+1.49, +5.30] |
| 控制 | 奖励选择器 | vs. 自评分 | +0.66 [-1.52, +2.92] |
| 控制 | 区域状态先验 | 恒等指标 | -7.28 [-13.33, -2.53] |
| 更新 | 随机更新 | vs. Ω₀ | +4.53 [+1.14, +9.17] |
| 更新 | 类型化更新 | vs. Ω₀ | +4.22 [+0.81, +8.30] |
| 更新 | 类型化更新 | vs. 随机更新 | -0.32 [-5.98, +5.15]（未解决） |
| 更新 | 无 fallback | vs. Base fallback | -0.773 [-1.471, -0.203] |
| 工具 | 检索 | vs. 无工具 | +0.562 [-0.700, +1.967]（未解决） |
| 工具 | 检索 + SAM | vs. 无工具 | +0.013 [-2.936, +3.263]（未解决） |
| 工具 | 增量 SAM | vs. 检索 | -0.548 [-3.390, +2.631]（未解决） |

**关键发现**: 三个控制轴均有独立贡献（文本 +5.62、噪声 +1.26、奖励 +3.33），但类型化更新相比随机更新无统计显著优势（-0.32，未解决）。移除 Base fallback 有显著负影响（-0.773）。区域状态先验实验失败（-7.28），说明局部区域控制对视频世界模型并不适用。

### Table 6: Selective Grounded-Router 最终结果

| 策略 | 卡片数 | Target | Confirmation | Pooled |
|------|--------|--------|--------------|--------|
| Exact no-tool | 0 | 0.4071 | 0.4425 | 0.4189 |
| Always-on profiled card | 180 | 0.4021 | 0.4352 | 0.4132 |
| Fit-only group policy | 48 | 0.4096 | 0.4401 | 0.4197 |
| Same-budget stratified random | 48 | 0.4084 | 0.4424 | 0.4197 |

**表格说明**: 选择性接地路由器对 fit-group 策略（仅 48 个卡片）与 same-budget 随机基线几乎等同（0.4197 vs 0.4197），fit 组优势在置信区间内未解决。

### Table 8: 协议与统计状态矩阵

| 设置 | 独立单元 | 评估角色 | 主要比较 |
|------|----------|----------|----------|
| GPT-5.4 SceneLang | 场景；3 固定种子平均 | 组件证据 | SceneLang vs Base；CI 条件于这些种子 |
| 学习型奖励 | 64 场景；179 嵌套视图 | 组件证据 | 场景分组 OOF 奖励 vs 匹配随机 |
| Physics-IQ 工具轴 | 场景；3 嵌套视图 | 预注册负消融 | Web-only 和 web+SAM vs 无工具；增量 SAM vs web-only |
| PAI split6 | 任务 | 同拆分部署诊断 | 所有匹配控制；SCOPE 事后分离 |
| OpenS2V | 类别分层条目 | 前瞻部署案例 | SCOPE vs LingBot；未解决 |
| Fresh PhyT2V 评估 | 场景 | 部署诊断 | 选中 vs 现役/固定提议；无扩展信用 |
| Fresh 32-task Physics-IQ 迁移 | 任务 | 前瞻集成测试 | 完整工具链 vs Base/路由器；冻结守护门 |
| 类型化更新确认 | 场景 | 前瞻集成测试 | 类型化更新 vs 匹配随机加伤害上限 |
| 接地选择性路由器 | 任务；角色/轮廓/家族自举 | 前瞻目标+确认 | fit 组 vs 无工具、始终启用和 same-budget 随机 |

**表格说明**: 完整的协议矩阵，区分了各实验设置的独立单元和评估角色，支持可重现性审计。

### Table 9: 完整 Official-72B PAI 质量画像

| 方法 | SC | BC | MS | AQ | IQ | OC | IS | IB |
|------|----|----|----|----|----|----|----|----|
| Frozen base (绝对值) | 89.51 | 91.71 | 98.49 | 44.21 | 62.83 | 19.50 | 92.44 | 95.75 |
| Shared global | -0.45 | -0.18 | -0.73 | +0.61 | +1.60 | -0.12 | -0.09 | -0.02 |
| Qwen-Image-Agent (SceneLang) | +0.89 | +0.37 | +0.13 | +0.42 | +0.50 | -0.06 | +0.66 | +0.49 |
| SceneLang + shared global | -0.03 | +0.19 | -0.65 | +0.79 | +2.25 | -0.04 | +0.10 | +0.03 |
| LingBot + safe fallback | +0.24 | +0.10 | +0.03 | 0.00 | +0.28 | -0.02 | +0.13 | +0.12 |
| LingBot Director→Pilot + shared global | -0.16 | +0.02 | -0.70 | +0.70 | +2.17 | -0.22 | +0.06 | +0.07 |
| LingBot Director→Pilot (raw) | +1.33 | +0.49 | +0.26 | -0.24 | +0.94 | +0.12 | +0.58 | +0.64 |
| LingBot + shared global | +0.15 | +0.22 | -0.55 | +0.60 | +1.73 | -0.26 | -0.08 | +0.15 |
| GPT-5.4-SKILL (raw) | -1.02 | -1.61 | +0.07 | -1.41 | -0.09 | +0.09 | +0.14 | +0.72 |
| GPT-5.4-SKILL + shared global | -2.67 | -2.27 | -0.66 | -1.64 | -1.14 | -0.24 | -0.54 | +0.07 |
| SCOPE w/o shared global | +0.81 | +0.36 | +0.13 | +0.41 | +0.57 | -0.01 | +0.61 | +0.44 |
| **SCOPE + shared global (ours)** | -0.06 | +0.20 | -0.68 | +0.69 | +2.33 | -0.03 | 0.00 | +0.02 |

**表格说明**: 在 8 个 PAI 质量维度上的详细分析。SCOPE w/o shared global 在多维均有正向提升，而 SCOPE + shared global 的表现与 SceneLang 独立相似，暗示 shared global 分量抵消了部分收益。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| [[Physics-IQ Verified|Physics-IQ]] | 40 场景 (main) / 32 (前瞻) | 物理推理视频生成基准 | 主评估（开发 + 持出） |
| [[PAI-Bench]] (PAI-Bench-G / split6) | 180 任务 | 通用视频质量评估 | 跨 backbone 诊断 |
| OpenS2V-Eval | 180 条目/backbone | 类别分层，多指标 | 前瞻部署案例 |
| PhyGround | 32 任务 | 物理合理性细粒度评估 | 前瞻测试（失败） |

### 实现细节

- **Backbone**: [[Wan2.2]]（主），[[CogVideoX]]（cross-backbone 验证）
- **开发轮数**: R=3（物理卡片 → 规则精化 → 最终 SCOPE 状态）
- **文本提议**: GPT-5.4 生成 SceneLang 物理概念卡，注入为文本条件
- **噪声提议**: 在固定调度基础上修改随机种子扩展候选集（M=3）
- **奖励选择**: OOF（out-of-fold）训练的学习型奖励，对候选排序
- **工具**: 网页检索（仅文本卡片）+ [[Grounded-SAM]]（分割验证）
- **统计方法**: 场景自举 [[Bootstrap Confidence Interval|95% 置信区间]]，双侧检验

### 可视化结果

- Figure 2 展示两个 backbone 上策略单调演化，R0→R3 性能持续提升
- Figure 3 展示检索增强生成在牛顿摆场景中的定性改善
- Figure 4 展示 SAM 辅助的区域状态干预（事后可视化）

---

## 批判性思考

### 优点

1. **形式化分数隔离**: 明确定义并形式化实现了分数隔离，是该领域目前最严格的评估协议之一，避免了隐性的测试集泄露
2. **可审计溯源链**: 密码学哈希绑定每次状态转移，完整的开发历史可独立验证，显著提升可重现性
3. **细致的统计报告**: 所有结果均报告 95% Bootstrap CI，区分"已解决"和"未解决"效果，科学诚实度高于多数 SOTA 论文

### 局限性

1. **选择可靠性不足**: 在未见任务上难以将候选质量转化为可靠的部署决策，前瞻评估中 15% 场景退化超过 10% 阈值
2. **Backbone 依赖性强**: shared-global 变体在 Wan (+0.19) 和 CogVideoX (-2.53) 间方向相反，效果无法跨模型迁移
3. **开发集代表性瓶颈**: 开发质量依赖于任务多样性；在同一账本上重复开发存在过拟合风险；PhyGround 完全失败（4/32 < 8/32 最低要求）
4. **类型化更新优势未得验证**: 消融实验中类型化更新 vs 随机更新的差异 (-0.32) 统计上未解决，核心设计假设缺乏明确实证支持

### 潜在改进方向

1. **自适应开发集构建**: 动态扩展开发集以覆盖更多物理场景类型，提升开发集代表性
2. **跨 backbone 适应策略**: 为不同架构设计条件化的控制状态，解决 backbone 依赖问题
3. **更强的提议质量估计器**: 在前瞻评估中提升候选质量预测的可靠性，降低场景退化率

### 可复现性评估

- [x] 代码开源（GitHub: https://github.com/YuhuaJiang2002/SCOPE）
- [ ] 预训练模型（未提及）
- [x] 训练细节完整（论文中有详细协议）
- [x] 数据集可获取（Physics-IQ 为公开基准）

---

## 关联笔记

### 基于

- [[Wan2.2]]: 主要 backbone，冻结视频世界模型
- [[CogVideoX]]: 第二 backbone，用于跨 backbone 验证
- [[Physics-IQ Verified|Physics-IQ]]: 主评估基准
- [[Grounded-SAM]]: 工具轴中的分割辅助组件

### 对比

- [[Director-Pilot]]: LingBot 风格的 Director→Pilot 代理方法，是主要对比 baseline 之一
- [[PAI-Bench]]: 通用视频质量评估基准，用于 cross-backbone 诊断

### 方法相关

- [[Score Isolation]]: SCOPE 的核心安全原则
- [[Inference-Time Adaptation]]: 框架的整体范式
- [[Provenance Binding]]: 可审计性机制
- [[Bootstrap Confidence Interval]]: 统计报告方法
- [[Video World Model]]: 被适配的目标模型类型

### 硬件/数据相关

- [[Physics-IQ Verified|Physics-IQ]]: 物理推理视频生成基准
- [[PAI-Bench]]: 通用视频质量多维评估

---

## 速查卡片

> [!summary] SCOPE: Score-Isolated Agentic Optimization for Video World Models
> - **核心**: 通过分数隔离原则对冻结视频世界模型进行可审计的推理时自适应
> - **方法**: 四轴类型化控制状态 + 提交-或-保留更新规则 + 密码学溯源绑定
> - **结果**: [[Physics-IQ Verified|Physics-IQ]] +14.24（Wan，95% CI [+8.10, +21.23]）；前瞻测试普遍未解决
> - **代码**: https://github.com/YuhuaJiang2002/SCOPE

---

*笔记创建时间: 2026-08-19*
