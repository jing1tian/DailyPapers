---
title: "CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents"
method_name: "CaPtain"
authors: [Yijia Fan, Xilin He, Yilei Jiang, Dongming Wu, Jiaxu Wang, Xiangyu Yue]
year: 2026
venue: arXiv
tags: [code-as-policy, robot-manipulation, reinforcement-learning, policy-learning, vision-language-model, post-training, embodied-ai]
zotero_collection: 3-Robotics/1-VLX/VLA
image_source: mixed
arxiv_html: https://arxiv.org/html/2610.40799
created: 2026-11-03
---

# 论文笔记：CaPtain: A Pipeline for Post-Training Code-as-Policy Robot Agents

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | CUHK MMLab, NovaXBot |
| 日期 | October 2026 |
| 项目主页 | — |
| 对比基线 | [[AgenticRoboRL]], [[ASPIRE]], [[K1]] |
| 链接 | [arXiv](https://arxiv.org/abs/2610.40799) / Code（随论文发布）|

---

## 一句话总结

> CaPtain 通过 SFT + 物理 Critic RL 两阶段后训练，将 Qwen3.5-4B 变成一个无需 VLA 或闭源模型的低级代码机器人智能体，LIBERO 成功率从 0% 提升至 73.8%。

---

## 核心贡献

1. **CaPtain-Harness**: 一个 ReAct 风格多轮代码执行框架，智能体用 Python 调用 22 个感知 + 运动原语，每次调用返回物理执行"收据"（receipt），无 VLA、无高级技能；
2. **两阶段后训练**：Stage 1 在合成轨迹上做 [[Supervised Fine-Tuning|SFT]]，Stage 2 用特权仿真器状态训练一个集成 MLP critic，通过 TD 逐请求分配物理信用（physical credit），驱动 [[Reinforcement Learning|RL]] 优化；
3. **开放性与泛化**：代码、数据和 CaPtain-4B 权重全部开源，实现跨仿真器零样本迁移（SimplerEnv 83.3%、CaP-Bench 75.0）和真实机器人部署。

---

## 问题背景

### 要解决的问题

[[Code-as-Policy]]（CaP）机器人智能体通常依赖闭源前沿模型（如 GPT-4）或外挂预训练 [[Vision-Language-Action|VLA]] 执行低级动作，测试时成本高且不开放。能否训练出一个小型开放 VLM，仅凭低级运动原语完成 LIBERO 级别的操作任务？

### 现有方法的局限

- **依赖前沿模型**：OpenRUA、ASPIRE、RACaP 等系统在测试时调用 GPT-4/Gemini，不可离线运行；
- **外挂 VLA**：Harness VLA、PyRUA、AgenticRoboRL 将 VLA 当作高级技能调用，冻结了低级控制的训练潜力；
- **单轮代码训练**：CaP-X 对 7B 编码器做单轮 GRPO，未覆盖多轮交互和低级运动原语。

### 本文的动机

代码作为策略天然支持感知-反馈循环和可解释调试；低级原语（IK、接触保护运动）让 VLM 直接控制关节而不需要 VLA；[[Reinforcement Learning|RL]] 中的物理 critic 可以在无法对语言 token 直接评奖时，从特权仿真器状态读取物理进度，精确信用分配给驱动成功的运动请求。

---

## 方法详解

### 模型架构

CaPtain 采用 **ReAct 多轮代码智能体** 架构：

- **输入**: 任务描述 $\ell$ + 当前 RGB 图像 $o_t$ + 所有历史动作与收据 $h_t$
- **Backbone**: [[Qwen|Qwen3.5-4B]]（指令微调版），视觉塔冻结，语言模型权重全量训练
- **核心模块**: [[Code-as-Policy]] 调用 22 个低级原语（感知、测量、受保护运动）
- **输出**: Python 代码块，harness 执行后返回文本收据
- **总参数**: ~4B
- **最大请求数**: 60 轮/episode

### 核心模块

#### 模块 1: CaPtain-Harness

**设计动机**: 复用 ReAct 循环，让 [[Vision-Language-Model|VLM]] 通过代码表达感知-运动闭环。

**具体实现**:
- 维护一个持久 Python 会话（persistent session），调用完毕后返回文本 receipt（物理发生了什么）
- 22 个原语包括：`mark_point`（感知）、`measure_distance`（测量）、`move_to`/`grasp`/`release`（受 [[Inverse Kinematics|IK]] 和接触保护 guarded contact 处理的运动）以及 `finish()`
- 没有 grasp/place 高级技能，没有 VLA，没有特权状态
- 支持多个后端仿真器（LIBERO/IsaacLab/RoboTwin/SimplerEnv）和真实机器人

#### 模块 2: 两阶段后训练

**Stage 1 — SFT**

用 GPT-6 Astra 在 harness 中生成合成轨迹，再对 [[Qwen|Qwen3.5-4B]] 做 [[Supervised Fine-Tuning|SFT]]：

- Loss 只作用于智能体自身生成的 token（不包括收据）
- 对反复失败 / 恢复循环的轮次令 $w_t=0$ 掩码
- 20% 概率用 mask token 替换早期 `mark_point` 坐标，迫使模型读当前图像
- VQA 样本权重 0.5

**Stage 2 — RL with Physical Critic**

集成 MLP critic 从特权仿真器状态预测任务成功率，再按 TD 差分给每个请求分配物理信用：

- critic 在训练时读取特权状态，测试时**完全不使用**
- 每轮：64 个起始状态 × 4 rollout = 256 episodes
- [[Generalized Advantage Estimation|GAE]] λ=0.9，无折扣；组相对 advantage 合并 episode 级别和请求级别排名
- KL 惩罚到 SFT 参考模型，防止退化
- token ratio mask $(0.2, 4)$：丢弃漂移过大的 token（而非截断）
- checkpoint: round 300（75 次优化步）

---

## 关键公式

### 公式 1: [[ReAct|请求历史]]

$$
h_t = (\ell,\, o_1,\, a_1,\, o_2,\, \dots,\, a_{t-1},\, o_t)
$$

**含义**: 时刻 $t$ 的策略输入，包含任务描述和所有历史动作/观测。

**符号说明**:
- $\ell$: 任务描述文本
- $o_t$: 当前 RGB 观测（含收据）
- $a_t$: 第 $t$ 轮生成的 Python 代码块

---

### 公式 2: [[Supervised Fine-Tuning|SFT 损失]]

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\sum_{t} w_t \sum_{j} \log \pi_\theta(a_{t,j} \mid h_t, a_{t,<j})
$$

**含义**: 只在智能体自身生成 token 上计算下一 token 预测损失，屏蔽失效轮次。

**符号说明**:
- $w_t$: 轮次权重，反复失败轮次令 $w_t=0$
- $a_{t,j}$: 第 $t$ 轮第 $j$ 个 token
- $\pi_\theta$: 待训练策略

---

### 公式 3: [[Reward Model|成功率 Critic]]

$$
V_\phi(s_t, t) = \sigma\!\left(\frac{1}{5}\sum_{k=1}^{5} f_{\phi_k}\bigl([x(s_t);\, t;\, e_{\text{task}}]\bigr)\right)
$$

**含义**: 集成 5 个 MLP 用特权仿真器状态预测 host success 概率。

**符号说明**:
- $x(s_t)$: 36 维特权状态特征（仅训练时可见）
- $t$: 请求时刻索引
- $e_{\text{task}}$: 任务 one-hot 编码
- $f_{\phi_k}$: 第 $k$ 个 MLP

---

### 公式 4: [[Reward Model|Critic 损失]]

$$
\mathcal{L}_V(\phi) = -\sum_i \sum_t \bigl[y_i \log V_\phi + (1-y_i)\log(1-V_\phi)\bigr]
$$

**含义**: 对每个 episode 的 host success 标签 $y_i$ 做二元交叉熵。

**符号说明**:
- $y_i \in \{0,1\}$: episode $i$ 的 host success 标签

---

### 公式 5: [[Temporal Difference|逐请求 TD 信用]]

$$
\delta_t = c_t + V_\phi(s_{t+1}, t) - V_\phi(s_t, t), \quad t < T
$$

$$
\delta_T = c_T + R_i - V_\phi(s_T, T)
$$

**含义**: 一次请求对预测成功率的变化量，固定时间索引以排除自然进展偏差；感知类调用获得接近零的信用。

**符号说明**:
- $c_t$: 每请求成本（$c_t \le 0$，激励效率）
- $R_i$: episode 终端奖励
- $T$: episode 总请求数

---

### 公式 6: [[Generalized Advantage Estimation|GAE]]

$$
A^{\mathrm{GAE}}_t = \sum_{k \ge 0} \lambda^k \delta_{t+k}, \quad \lambda = 0.9
$$

**含义**: 将 TD 信用向后传播，让早期感知请求能间接获得后续运动带来的信用。

**符号说明**:
- $\lambda=0.9$: GAE 衰减系数

---

### 公式 7: [[Group Relative Policy Optimization|组相对 Advantage]]

$$
A_{i,t} = \bigl(G_i - \bar{G}_{\text{group}}\bigr) + \bigl(A^{\mathrm{GAE}}_{i,t} - \overline{A^{\mathrm{GAE}}_{i,\cdot}}\bigr)
$$

$$
G_i = R_i + \sum_t c_{i,t}
$$

**含义**: 将 episode 级别（组间排名）和请求级别（episode 内排名）的优势合并，稳定训练。

**符号说明**:
- $G_i$: episode 总回报
- $\bar{G}_{\text{group}}$: 同一起始状态组的平均回报

---

### 公式 8: 终端奖励

$$
R_i = \begin{cases}
1.0 & \text{success + finish()} \\
0.25 & \text{success without finish()} \\
0.0 & \text{other failure} \\
-0.1 & \text{budget exhausted} \\
-0.3 & \text{false finish}
\end{cases}
$$

**含义**: 区分"可部署完成"（显式调用 finish）与"未声明成功"，惩罚预算耗尽和误报结束。

---

### 公式 9: [[Proximal Policy Optimization|策略损失]]

$$
\mathcal{L}_\pi(\theta) = -\sum_{i,t,j} m_{t,j}\, r_{t,j}\, A_{i,t} + \beta\, \mathrm{KL}_{k3}(\pi_\theta \| \pi_{\mathrm{SFT}})
$$

**含义**: 重要性加权策略梯度 + KL 惩罚到 SFT 参考策略，替代 PPO clip 的 token 级 ratio 掩码。

**符号说明**:
- $r_{t,j} = \pi_\theta(a_{t,j}\mid\cdot)/\pi_{\text{old}}(a_{t,j}\mid\cdot)$: token 级 IS 比率
- $m_{t,j} = \mathbf{1}[0.2 < r_{t,j} < 4]$: ratio 漂移掩码（丢弃而非截断）
- $\beta$: KL 惩罚系数
- $\mathrm{KL}_{k3}$: k3 估计量

---

## 关键图表

### Figure 1: CaPtain 系统概览

![[CaPtain_fig1.png|600]]

**说明**: 上半部分展示 CaPtain-Harness 推理流程：VLM 生成 Python 代码 → harness 执行 → 返回 receipt；下半部分展示训练与测试流程，物理 critic 仅在训练时访问特权仿真器状态。

---

### Figure 2: RL 增益跨扰动稳健性

*（图片 URL 未在 arXiv HTML 中暴露）*

**说明**: SFT vs. RL 在 held-out LIBERO 和四个扰动 suite（LIBERO-PRO/Plus/MAX）上的 host success 和 strict success 对比，展示 RL 增益在分布外扰动下的保持能力。

---

### Figure 3: CaPtain-4B-SFT vs. CaPtain-4B 细粒度拆分

*（图片 URL 未在 arXiv HTML 中暴露）*

三个子图：
- **(a)** LIBERO 按 suite 分 strict success
- **(b)** LIBERO-Plus 按 7 个扰动维度分 host success
- **(c)** LIBERO-PRO 按 suite 和扰动类型分 host success

---

### Figure 4: 真实机器人演示

![[CaPtain_fig2.png|600]]

**说明**: Astrabot XR1 臂在真实场景（新物体/布局）完成两个任务，policy 仅用仿真器双胞胎（simulated twin）适配，无真实世界数据。

---

### Figure 5: 仿真双胞胎 vs. 真实场景

![[CaPtain_fig3.png|600]]

**说明**: 左侧为用于 XR1 臂适配的粗糙仿真双胞胎，右侧为真实测试场景，展示 sim-to-real 迁移的外观差距。

---

### Figure 6: CaPtain-Harness 详细设计

![[CaPtain_fig4.png|600]]

**说明**: 展示持久 Python 会话架构、22 个低级原语分类（感知/测量/受保护运动）、IK 和 guarded contact 细节、receipt 格式、completion gate，以及多后端支持。

---

### Figure 7: Finish Collapse 分析

*（图片 URL 未在 arXiv HTML 中暴露）*

**说明**: 在 RL round 340 后 host success 保持稳定，但 strict success 下降（finish collapse）；两组测试集（40 official seed-11 + 40 wide-layout）均可观察。

---

### Figure 8: RoboTwin 2.0 零样本迁移演示

![[CaPtain_fig5.png|600]]

**说明**: 双 Franka 臂在随机场景中，CaPtain-4B 抓杯缘、沿测量偏移搬到垫子、松开。无 LIBERO 以外训练数据。

---

### Figure 9: RoboDojo 零样本迁移演示

![[CaPtain_fig6.png|600]]

**说明**: 双 ARX X5 臂堆叠三个方块，通过举起、放置和重新抓取完成。主要失败原因为臂部伸展限制。

---

### Table 1: LIBERO 主要结果（Host Success）

| 方法 | LIBERO | LIBERO-PRO (3) | LIBERO-Plus | LIBERO-MAX Dyn |
|------|--------|----------------|-------------|----------------|
| Qwen3.5-4B (zero-shot) | ~0% | ~0% | ~0% | ~0% |
| CaPtain-4B-SFT | 55.4% | 59.5% | 48.6% | 40.0% |
| **CaPtain-4B** | **73.8%** | **78.5%** | **66.1%** | **61.2%** |
| AgenticRoboRL | — | — | — | — |
| K1 | — | 77.2% | — | — |
| WAA | — | 75.6% | — | — |

**关键发现**: RL 在所有 LIBERO 扰动 suite 上带来一致 ~15-19% 的绝对提升（vs. SFT），且比 AgenticRoboRL 的 +5.9% RL 增益高出 3 倍以上。

---

### Table 2: 零样本迁移结果

| Benchmark | 成功率 | 说明 |
|-----------|--------|------|
| SimplerEnv WidowX | 83.3% | 4 任务 × 24 episodes |
| CaP-Bench | 75.0 | 优于 Gemini-3-Pro (73.5) |
| RoboTwin 2.0 Hard | 22.6% (总体) / 30.0% (单臂) | 双臂协调难度大 |
| RoboDojo | 0.8% success / score 1.7 | 主要受限于臂部伸展 |

---

### Table 3: Critic 信用分布（优势质量分析）

| 请求类型 | 优势质量占比 |
|----------|-------------|
| 运动类（move/grasp/release） | ~65% + 14% + 14% = **93%** |
| 感知类（perception） | **1.3%** |

**关键发现**: critic 将 93% 的信用分配给实际驱动物理变化的运动请求，仅 1.3% 到感知请求，验证了物理 critic 的精准信用分配能力。

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| LIBERO | 40 任务 × 50 demo | 多任务桌面操作 | SFT + RL 训练 |
| LIBERO (held-out) | 240 episodes | initial states 11-16 | 测试 |
| LIBERO-PRO | 600/800 episodes | 3/4 扰动 suite | 鲁棒性测试 |
| LIBERO-Plus | 700 cases | 7 扰动维度 | 鲁棒性测试 |
| LIBERO-MAX | 80×2 cases | Base/Dynamic | 极端扰动 |
| SimplerEnv WidowX | 4×24 episodes | 零样本 | 迁移测试 |
| RoboTwin 2.0 Hard | 5,000 episodes | 双臂/随机场景 | 迁移测试 |
| CaP-Bench | Cube Lift/Stack | 代码策略 benchmark | 迁移测试 |

### 实现细节

- **Base model**: Qwen3.5-4B（instruction-tuned，视觉塔冻结）
- **SFT 数据**: GPT-6 Astra 在 CaPtain-Harness 中合成的轨迹
- **RL 轮次**: 每轮 256 episodes（64 起始状态 × 4 rollout），每 4 轮 1 个优化步
- **Critic**: 5 MLP × 3 隐层（256 单元），在线共训练（每轮 2 次 AdamW 步）
- **推理**: vLLM，temperature=1，top-p=1，thinking 开启，最长 4,096 token/请求
- **硬件**: 8 个 episode 共享 1 GPU；每 episode 约 52 GPU-s，成本 $0.03–0.06
- **RL checkpoint**: round 300（75 优化步）

### 可视化结果

- RL 在 LIBERO 5 个 suite 上一致提升，且在分布外扰动（lighting、camera、materials、layout）下均保持增益
- Finish collapse 在 round 340 后出现：host success 稳定但 strict success 下降，是多目标 RL 中的已知现象
- 真实机器人任务通过纯仿真适配完成，展示 sim-to-real 可行性

---

## 批判性思考

### 优点

1. **完全开放**: 代码、数据、模型权重全部开源，可复现性高；
2. **无 VLA 依赖**: 测试时仅用 4B 模型 + 低级原语，部署成本极低；
3. **精准信用分配**: 物理 critic 有效避免感知类 token 获得虚高奖励；
4. **扰动鲁棒性**: RL 增益在 LIBERO-PRO/Plus/MAX 上均稳定转移，非过拟合。

### 局限性

1. **单臂 LIBERO 训练**: RL 仅在 LIBERO 单臂任务上收敛，双臂协调（RoboDojo 0.8%）还很弱；
2. **Critic 依赖精确仿真器状态**: 真实场景的姿态噪声会削弱信用信号；
3. **Finish Collapse**: RL 训练晚期 strict success 下降，说明多目标平衡仍是挑战；
4. **力顺应任务缺失**: Wipe 等需要力反馈的任务无法用当前原语表达；
5. **感知服务未计入成本**: 实际部署成本比 $0.03–0.06/episode 更高。

### 潜在改进方向

1. 扩展原语集合支持力反馈/顺应控制；
2. 用真实机器人数据 fine-tune 解决 finish collapse；
3. 设计不依赖精确仿真器状态的 critic（如视觉 critic）。

### 可复现性评估

- [x] 代码开源
- [x] 预训练模型（CaPtain-4B 权重）
- [x] 训练细节完整（部分 Appendix 可获取）
- [x] 数据集可获取（合成 SFT 数据随论文发布）

---

## 关联笔记

### 基于

- [[Qwen]]: 使用 Qwen3.5-4B 作为基座模型
- [[Code-as-Policy]]: CaPtain 的核心框架
- [[ReAct]]: 多轮代码执行的 agent 范式
- [[Generalized Advantage Estimation]]: RL 信用分配

### 对比

- [[AgenticRoboRL]]: 同为 RL 训练的代码智能体，CaPtain RL 增益 3× 更大
- [[ASPIRE]]: 用闭源 API 的 CaP 方法
- [[K1]]: LIBERO-PRO 基线（77.2%）
- [[VLA]]: 端到端策略，LIBERO ~94-98% 但扰动脆弱

### 方法相关

- [[Reinforcement Learning]]: Stage 2 训练范式
- [[Supervised Fine-Tuning]]: Stage 1 训练范式
- [[Reward Model]]: 物理 critic 的角色
- [[Group Relative Policy Optimization]]: 组相对 advantage 设计参考
- [[Inverse Kinematics]]: harness 低级运动原语
- [[KL Divergence]]: 策略正则化

### 硬件/数据相关

- [[LIBERO]]: 主要训练和评测 benchmark
- [[SimplerEnv]]: 零样本迁移 benchmark
- [[IsaacLab]]: 仿真后端之一

---

## 速查卡片

> [!summary] CaPtain (arXiv 2026)
> - **核心**: SFT + 物理 Critic RL 后训练 4B VLM 做 Code-as-Policy 机器人智能体
> - **方法**: CaPtain-Harness（22 个低级原语）+ 特权 MLP Critic 的 TD 信用分配
> - **结果**: LIBERO 73.8%（SFT 55.4%），SimplerEnv 83.3%，CaP-Bench 75.0
> - **代码**: 随 arXiv:2610.40799 发布

---

*笔记创建时间: 2026-11-03*
