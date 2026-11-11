---
title: "DreamVLA: From Generated Human Motions to Humanoid Vision-Language-Action Policies"
method_name: "DreamVLA"
authors: [Kevin Park, Izat Rasyad, Dvij Kalaria, Shankar Sastry]
year: 2026
venue: CoRL 2026 Workshop (G2A + Open Problems in Contact-Rich Loco-Manipulation)
tags: [vla, humanoid, whole-body-control, imitation-learning, reinforcement-learning, human-motion-generation, dagger, loco-manipulation]
zotero_collection: Robotics/VLA
image_source: online
arxiv_html: https://arxiv.org/html/2611.10872
created: 2026-11-11
---

# 论文笔记：DreamVLA: From Generated Human Motions to Humanoid Vision-Language-Action Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | Princeton University, University of California Berkeley |
| 日期 | November 2026 |
| 项目主页 | https://ember-center-berkeley.github.io/DreamVLA/ |
| 对比基线 | [[SONIC]], [[DAgger]] (Ross et al.) |
| 链接 | [arXiv](https://arxiv.org/abs/2611.10872) / Code: N/A |

---

## 一句话总结

> DreamVLA 通过生成人体运动轨迹引导仿真 RL 专家，再以参考条件 DAgger 蒸馏为 GR00T VLA 策略，无需新的遥操作数据即可完成人形机器人全身操作任务。

---

## 核心贡献

1. **生成式演示流水线**: 用 [[OmniControl]] 生成受时空关节约束的人体运动，经 [[OmniRetarget]] 重定向到 Unitree G1，再用 [[HumanoidMimicGen]] 增强接触点，从而在仿真中无需人工设计奖励即可生成任务参考轨迹
2. **残差 RL 专家**: 在 [[SONIC]] 名义 token 之上叠加可学习的残差修正 $\Delta z_t$，使 RL 专家在保持自然运动的同时显著提升任务成功率（Picking: 0% → 88.6%）
3. **参考条件 Recovery DAgger**: 对失败的 VLA 学员 rollout，从随机采样的失败状态以相同参考后缀重启专家，只聚合成功的恢复片段，解决标准 DAgger 在接触精细任务上的分布偏移问题

---

## 问题背景

### 要解决的问题

[[视觉语言动作模型|VLA]] 策略通常需要大量机器人演示数据，对人形机器人的全身操作任务进行遥操作采集成本极高。如何在不采集新遥操作数据的前提下，训练能完成全身操作的 [[VLA]] 策略？

### 现有方法的局限

- **稀疏奖励 RL**: 人形机器人操作任务奖励稀疏，RL 探索困难
- **密集奖励 RL**: 需要为每个任务手工设计奖励函数，难以泛化
- **离线模仿学习**: 存在分布偏移，小误差会不断累积，尤其在需要精确接触的任务（如抓取、按钮）中学员进入训练数据从未覆盖的状态
- **标准 DAgger**: 在接触敏感操作中专家从失败学员状态出发的修复动作可能不可靠或过于突兀

### 本文的动机

生成的人体运动天然包含全身协调的结构性先验，[[DreamControl]] 已证明其可引导 RL。在此基础上，DreamVLA 完整地解决了从参考生成到 VLA 蒸馏的全流水线，并用 Recovery DAgger 处理精细接触任务中的分布偏移。

---

## 方法详解

### 模型架构

DreamVLA 是一个四阶段流水线：

1. **参考生成** (Reference Generation): [[OmniControl]] 采样满足时空关节约束的人体运动，[[OmniRetarget]] 重定向到 G1，[[HumanoidMimicGen]] 增强接触点
2. **残差 RL 专家训练** (Residual Expert): 在 [[SONIC]] 名义 token 上训练残差修正策略 $\pi_{\mathrm{res}}$，使用混合奖励
3. **数据集收集** (Dataset Collection): 成功 rollout 以 LeRobot 格式导出；VLA 学员只看自中心相机图像 + 本体感知 + 语言指令
4. **GR00T 蒸馏 + Recovery DAgger** (Distillation): [[GR00T N1]] offline 初始化后，迭代式 Recovery DAgger 聚合修复片段

- **硬件平台**: Unitree G1（仅仿真，无真实部署）
- **VLA 骨干**: 预训练 [[GR00T N1]] (fine-tuned)
- **底层控制器**: [[SONIC]] 全身运动追踪策略

### 核心模块

#### 模块1: 参考运动生成

**设计动机**: 利用 [[OmniControl]] 的时空关节约束生成任务相关的人体运动，自动提供全身协调先验，取代手工设计奖励

**具体实现**:
- [[OmniControl]] 以语言指令 $\ell$ 和时空关节约束集合 $\mathcal{G}$ 为条件采样人体运动序列 $H$
- 约束 $\mathcal{G} = \{(j_k, \mathbf{v}_k, t_k)\}_{k=1}^{K}$，其中 $j_k$ 为关节，$\mathbf{v}_k$ 为目标值，$t_k$ 为时刻
- 人体运动经 [[OmniRetarget]] 重定向到 Unitree G1 的关节配置
- [[HumanoidMimicGen]] 进一步修改接触点，使参考轨迹更适合任务

#### 模块2: 残差 RL 专家

**设计动机**: [[SONIC]] 名义 token 在操作任务中精度不足（Pick 0%），直接对 token 空间学习绝对值会导致追踪失败（MPJPE 62.29mm），残差修正保留了运动自然性同时提升任务成功率

**具体实现**:
- 名义 token: $z_t^{\mathrm{nom}} = E_{\mathrm{S}}(R_{t:t+F})$（SONIC 编码器根据参考窗口生成）
- 残差修正: $\Delta z_t = \pi_{\mathrm{res}}(s_t, R_{t:t+F}, \bar{x}_{t:t+F}, \ell)$
- 合并: $z_t^{\mathrm{E}} = z_t^{\mathrm{nom}} + \Delta z_t$，经 SONIC 解码器得到关节目标 $a_t^{\mathrm{body}}$
- 额外输出: 二值手部开合命令
- 训练时使用物理域随机化；VLA 训练时不使用视觉随机化
- 专家观测包含特权物体状态；VLA 学员只看视觉和本体感知

#### 模块3: Recovery DAgger

**设计动机**: 标准 DAgger 让专家在学员失败状态重新开始，但在精细接触任务中该起点的专家修复动作可能不可靠；Recovery DAgger 只保留最终成功的完整修复片段

**具体实现**:
1. 对每条失败的学员 rollout，随机采样一个时刻 $t$，恢复仿真器到该时刻的状态
2. 从恢复状态以相同参考后缀重启 RL 专家
3. 只有专家在剩余参考窗口内成功完成任务的恢复轨迹 $\widetilde{\tau}_{i,t}$ 才被聚合
4. 动作块在接受的恢复片段内构建，不跨 episode 或恢复边界

---

## 关键公式

### 公式1: [[OmniControl|参考运动采样]]

$$
H \sim p_{\mathrm{motion}}(H \mid \ell, \mathcal{G}), \qquad \mathcal{G} = \{(j_k, \mathbf{v}_k, t_k)\}_{k=1}^{K}
$$

**含义**: 以语言指令 $\ell$ 和时空关节约束集合 $\mathcal{G}$ 为条件，从 OmniControl 中采样满足约束的人体运动序列 $H$

**符号说明**:
- $H$: 生成的人体运动序列
- $\ell$: 语言任务指令
- $\mathcal{G}$: 时空关节约束集合
- $j_k, \mathbf{v}_k, t_k$: 第 $k$ 个约束的关节、目标值、时刻

---

### 公式2: 参考观测窗口

$$
\mathcal{O}_t^{\mathrm{ref}} = [\gamma_{t|t}, \gamma_{t+\Delta t_{\mathrm{obs}}|t}, \ldots, \gamma_{t+F|t}]
$$

**含义**: 专家在 $t$ 时刻观测一个以 0.1 秒间隔的未来参考帧窗口

**符号说明**:
- $\mathcal{O}_t^{\mathrm{ref}}$: $t$ 时刻的参考观测窗口
- $\gamma_{\tau|t}$: 在 $t$ 时刻观测的 $\tau$ 时刻目标状态（见公式3）
- $\Delta t_{\mathrm{obs}} = 0.1\,\mathrm{s}$: 参考帧时间间隔
- $F$: 窗口长度

---

### 公式3: 目标状态向量

$$
\gamma_{\tau|t} = (q_\tau^{\mathrm{ref}}, \dot{q}_\tau^{\mathrm{ref}}, p_{\tau|t}^{\mathrm{rel,root}}, p_{\tau|t}^{\mathrm{rel,key}}, s_\tau^{\mathrm{ref,left}}, s_\tau^{\mathrm{ref,right}})
$$

**含义**: 参考目标状态包含关节角/速度（$\mathbb{R}^{27}$）、根部相对姿态、关键点相对位置（$\mathbb{R}^{3 \times 41}$）和双手状态

**符号说明**:
- $q_\tau^{\mathrm{ref}}, \dot{q}_\tau^{\mathrm{ref}} \in \mathbb{R}^{27}$: 参考关节角度和速度
- $p_{\tau|t}^{\mathrm{rel,root}}$: 相对根部的参考姿态
- $p_{\tau|t}^{\mathrm{rel,key}} \in \mathbb{R}^{3 \times 41}$: 在机器人坐标系下的关键点位置
- $s_\tau^{\mathrm{ref,left/right}}$: 左右手参考状态

---

### 公式4: 关键点坐标变换

$$
p_{\tau|t}^{\mathrm{rel,key},j} = Q_t^{\mathsf{T}} \left( p_\tau^{\mathrm{ref,key},j} - p_t^{\mathrm{robot,root}} \right)
$$

**含义**: 将参考关键点从世界坐标系变换到机器人根部坐标系，使参考观测具有平移不变性

**符号说明**:
- $Q_t$: $t$ 时刻机器人根部的旋转矩阵
- $p_\tau^{\mathrm{ref,key},j}$: 世界坐标系下第 $j$ 个参考关键点位置
- $p_t^{\mathrm{robot,root}}$: 机器人根部位置

---

### 公式5: [[Residual Policy|残差 Token 计算]]

$$
z_t^{\mathrm{nom}} = E_{\mathrm{S}}(R_{t:t+F})
$$

$$
\Delta z_t = \pi_{\mathrm{res}}(s_t, R_{t:t+F}, \bar{x}_{t:t+F}, \ell)
$$

$$
z_t^{\mathrm{E}} = z_t^{\mathrm{nom}} + \Delta z_t
$$

**含义**: 专家 token 由 SONIC 名义 token 加上残差修正构成，残差修正由策略网络 $\pi_{\mathrm{res}}$ 根据当前状态、参考窗口、物体参考和语言指令计算

**符号说明**:
- $z_t^{\mathrm{nom}}$: SONIC 编码器生成的名义 token
- $E_{\mathrm{S}}$: SONIC 编码器
- $\Delta z_t$: 学习得到的残差修正
- $\pi_{\mathrm{res}}$: 残差策略网络
- $s_t$: 当前机器人状态
- $\bar{x}_{t:t+F}$: 物体参考状态
- $z_t^{\mathrm{E}}$: 最终专家 token

---

### 公式6: 体态动作解码

$$
a_t^{\mathrm{body}} = D_{\mathrm{S}}(p_t, z_t^{\mathrm{E}})
$$

**含义**: SONIC 解码器将专家 token 和当前本体感知状态解码为关节角度目标

**符号说明**:
- $D_{\mathrm{S}}$: SONIC 解码器
- $p_t$: 当前本体感知状态
- $a_t^{\mathrm{body}}$: 输出的体态关节指令

---

### 公式7: [[奖励函数|专家奖励函数]]

$$
r_t = w_{\mathrm{hum}} r_{\mathrm{hum}}(s_t, R_t) + w_{\mathrm{task}} r_{\mathrm{task}}(s_t) - w_{\mathrm{reg}} c_{\mathrm{reg}}(u_t^{\mathrm{E}}, u_{t-1}^{\mathrm{E}})
$$

**含义**: 混合奖励由人体运动追踪项、任务稀疏奖励项和控制正则化惩罚项构成，平衡运动自然性与任务完成

**符号说明**:
- $r_{\mathrm{hum}}$: 人体运动追踪奖励（鼓励跟随参考）
- $r_{\mathrm{task}}$: 任务稀疏奖励（成功时给予）
- $c_{\mathrm{reg}}$: 控制正则化惩罚（抑制急动）
- $w_{\mathrm{hum}}, w_{\mathrm{task}}, w_{\mathrm{reg}}$: 权重系数
- $u_t^{\mathrm{E}}, u_{t-1}^{\mathrm{E}}$: 当前和前一时刻的控制输入

---

### 公式8: 初始数据集（成功 rollout）

$$
\mathcal{D}_0 = \bigcup_{\tau:\, \mathsf{S}(\tau)=1} \{(o_t, \ell, u_t^{\mathrm{E}})\}_{t=0}^{T_\tau - 1}
$$

**含义**: 初始数据集只包含成功 rollout 的观测-动作对，$\mathsf{S}(\tau)=1$ 表示任务成功

**符号说明**:
- $\mathsf{S}(\tau)$: 轨迹 $\tau$ 的成功标志
- $o_t$: 视觉+本体感知观测
- $u_t^{\mathrm{E}}$: 专家动作块

---

### 公式9: [[行为克隆|GR00T 离线目标]]

$$
\theta_0 = \arg\min_{\theta} \mathbb{E}_{(o_t, \ell, U_t^{\mathrm{E}}) \sim \mathcal{D}_0} \left[ \mathcal{L}_{\mathrm{GR00T}}(\theta; o_t, \ell, U_t^{\mathrm{E}}) \right]
$$

**含义**: 在初始数据集上对 GR00T 进行离线行为克隆，最小化 GR00T 自身的动作预测损失

**符号说明**:
- $\theta$: GR00T 模型参数
- $\mathcal{L}_{\mathrm{GR00T}}$: GR00T 预训练动作损失
- $U_t^{\mathrm{E}}$: 专家动作块

---

### 公式10: [[DAgger|标准 DAgger 聚合]]

$$
\mathcal{D}_{k+1} = \mathcal{D}_k \cup \{(o_t, \ell, \pi_{\mathrm{E}}(s_t, c_t)) : s_t \in \mathcal{V}_k\}
$$

**含义**: 标准 DAgger 在学员访问过的状态集合 $\mathcal{V}_k$ 上查询专家，将专家标注加入数据集

**符号说明**:
- $\mathcal{V}_k$: 第 $k$ 轮学员访问的状态集合
- $\pi_{\mathrm{E}}$: 专家策略
- $c_t$: 专家条件（参考后缀等）

---

### 公式11: [[DAgger|参考条件 Recovery DAgger 聚合]]

$$
\mathcal{D}_{k+1} = \mathcal{D}_k \cup \bigcup_{\widetilde{\tau}_{i,t}:\, \mathsf{S}(\widetilde{\tau}_{i,t})=1} \{(\widetilde{o}_j, \ell, \widetilde{u}_j^{\mathrm{E}})\}_{j=t}^{T^{\mathrm{rec}}_{i,t}-1}
$$

**含义**: Recovery DAgger 只聚合从失败学员状态出发、以相同参考后缀重启专家后**成功完成任务**的恢复片段，动作块不跨 episode 边界

**符号说明**:
- $\widetilde{\tau}_{i,t}$: 从第 $i$ 条失败轨迹的 $t$ 时刻恢复的轨迹
- $\mathsf{S}(\widetilde{\tau}_{i,t})$: 恢复轨迹是否成功
- $\widetilde{o}_j, \widetilde{u}_j^{\mathrm{E}}$: 恢复片段中的观测和专家动作
- $T^{\mathrm{rec}}_{i,t}$: 恢复片段结束时刻

---

## 关键图表

### Figure 1: 仿真任务场景

![Figure 1 - DreamVLA Task Scenes](https://arxiv.org/html/2611.10872v1/dreamvla-task-scenes.png)

**说明**: DreamVLA 的四个仿真任务场景（Unitree G1）：(a) 从平台抓取物体、(b) 开抽屉、(c) 按按钮、(d) 行走后从平台抓取物体（Loco-Manipulation）。图片展示了任务的初始状态配置。

---

### Figure 2: 名义 SONIC vs 残差控制对比

![Figure 2 - DreamVLA Residual Comparison](https://arxiv.org/html/2611.10872v1/dreamvla-residual-comparison.png)

**说明**: 在芥末瓶抓取任务中，(a) 名义 [[SONIC]] 零样本追踪将瓶子推落台面（帧 50/250/739，对应 1.00s/5.00s/14.78s）；(b) 残差控制将瓶子保持在手中。展示了残差 RL 在接触精细操作中的关键优势。

---

### Table 1: DAgger 迭代次数消融（每格 100 次试验）

| 变体 | DAgger 轮数 | 轨迹总数 | Pick | Drawer | Button | Walk & Pick |
|------|------------|---------|------|--------|--------|-------------|
| Offline | 0 | 200 | 0% | 51% | 59% | 0% |
| DAgger | 1 | 400 | 17% | 55% | 58% | 1% |
| DAgger | 2 | 600 | 30% | 54% | 62% | 7% |
| DAgger | 3 | 800 | 54% | 60% | 69% | 18% |
| Matched Offline | 0 | 800 | 0% | 54% | 57% | 0% |

**关键发现**: Matched Offline（同等数据量但无恢复聚合）在 Pick 和 Walk & Pick 上保持 0%，证明提升来自纠正性聚合而非数据量扩充。3 轮 DAgger 后 Pick 从 0% 提升至 54%。

---

### Table 2: 残差 RL vs 名义 SONIC（3000 次迭代，专家评估）

| 方法 | Pick | Drawer | Button | Walk & Pick | [[MPJPE]] (mm) |
|------|------|--------|--------|-------------|----------------|
| 名义 SONIC | 0% | 30.0% | 68.0% | 17.30% | 24.84 |
| **残差 Token RL（本文）** | **88.6%** | **74.0%** | **81.0%** | **75.95%** | 25.51 |
| 绝对 Token RL | 0% | 0% | 0% | 0% | 62.29 |

**关键发现**: 残差 RL 以极小的追踪精度损失（+0.67mm [[MPJPE]]）换取巨大的任务成功率提升；绝对 token 方案追踪精度极差（62.29mm），任务全部失败。

---

### Table 3: 生成数据规模消融（参考数 = episode 数）

| 参考数量 | Pick | Drawer | Button | Walk & Pick |
|---------|------|--------|--------|-------------|
| 10 | 0% | 13% | 0% | 0% |
| 50 | 0% | 51% | 14% | 0% |
| 200 | 0% | 51% | 59% | 0% |

**关键发现**: Drawer 成功率在 50-200 个参考之间趋于饱和，Button 持续提升；Pick 和 Walk & Pick 在此消融中始终为 0%（文中未充分解释，可能是离线初始化阶段的瓶颈）。

---

### Table 4: 任务成功标准

| 任务 | 成功判定 |
|------|---------|
| Pick from Platform | 物体被抬升 ≥15 cm 并保持 |
| Open Drawer | 抽屉位移 >0.15 m |
| Press Button | 右手食指接触按钮 |
| Walk and Pick | 走到平台后抬升物体 ≥15 cm |

---

## 实验

### 数据集

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| Generated Expert Trajectories | 200 条（初始），每轮 +200 条（Recovery DAgger） | 仿真专家成功 rollout，LeRobot 格式 | VLA 训练（$\mathcal{D}_0$） |
| Recovery Segments | 每轮约 200 条 | 专家从失败学员状态恢复成功的片段 | DAgger 聚合 |
| OmniControl References | 10 / 50 / 200（消融） | 生成的人体运动参考轨迹 | 引导 RL 专家 |

### 实现细节

- **Backbone**: 预训练 GR00T N1 系列（具体变体未说明）
- **底层控制器**: [[SONIC]]（Unitree G1）
- **参考生成**: [[OmniControl]] + [[OmniRetarget]] + [[HumanoidMimicGen]]
- **RL 迭代**: 3000 次（残差/绝对 token 对比实验）
- **状态维度**: 关节 $q, \dot{q} \in \mathbb{R}^{27}$；关键点 $\in \mathbb{R}^{3 \times 41}$；$\Delta t_{\mathrm{obs}} = 0.1$s
- **手部**: 二值开合命令
- **域随机化**: 物理随机化用于专家训练；VLA 训练不使用视觉随机化
- **硬件**: 仅仿真（Unitree G1），无真实部署
- **评估**: 每格 100 次试验

### 可视化结果

残差 RL 专家能在抓取芥末瓶时维持稳定接触；名义 SONIC 在同一任务中将瓶子推落，说明在接触精细任务中名义追踪的精度不足，需要残差修正。

---

## 批判性思考

### 优点

1. **数据效率高**: 无需新的遥操作数据，从人体运动生成到 VLA 策略的端到端流水线可扩展
2. **Recovery DAgger 设计合理**: 只聚合成功恢复片段有效避免了错误示范的污染，matched offline 对照实验有力支持了方法有效性
3. **残差设计保留动作自然性**: 在 SONIC token 空间做残差修正，MPJPE 仅增加 0.67mm，同时任务成功率大幅提升

### 局限性

1. **仅四个仿真任务，无真实部署**: 泛化性未在更多任务或真实 G1 上验证
2. **生成参考可能不可行**: OmniControl 生成的运动可能在运动学或接触上不可行，需要后续过滤和奖励调整
3. **数据规模消融存在疑点**: Pick 和 Walk & Pick 在数据规模消融中始终为 0%，可能说明离线初始化本身是瓶颈
4. **专家修复不可靠**: 在精细接触操作时，专家从学员失败状态出发的修复动作可能突变或不可靠

### 潜在改进方向

1. 在真实 Unitree G1 上进行 sim-to-real 迁移验证
2. 扩展到更多任务类型和物体，测试流水线的泛化性
3. 研究为什么 Pick 在离线阶段始终失败，可能需要更好的离线初始化策略
4. 将 Recovery DAgger 与更先进的在线 RL 蒸馏（VIRAL 等）结合

### 可复现性评估

- [ ] 代码开源（未公开）
- [ ] 预训练模型（未公开）
- [x] 训练细节（主要参数在论文中描述）
- [ ] 数据集（生成数据流程依赖专有工具链）

---

## 关联笔记

### 基于

- [[DreamControl]]: 本文直接基于 DreamControl 的参考生成和奖励设计，DreamVLA 是其 VLA 蒸馏扩展
- [[SONIC]]: 底层全身运动追踪控制器，DreamVLA 在其 token 空间上叠加残差 RL
- [[GR00T N1]]: VLA 骨干，DreamVLA 对其进行任务特定微调

### 对比

- [[DAgger]]: 标准 DAgger (Ross et al. 2011) 作为方法基线
- [[SONIC]]: 名义 SONIC 追踪 vs 残差 RL 专家的直接对比

### 方法相关

- [[OmniControl]]: 参考运动生成的核心工具，时空约束人体运动生成
- [[OmniRetarget]]: 人体运动到 G1 的运动重定向
- [[HumanoidMimicGen]]: 接触点增强
- [[Residual Policy]]: 残差策略设计的核心概念
- [[Action Chunking]]: VLA 策略的动作输出形式
- [[Whole-Body Control]]: 任务所需的全身协调控制
- [[Loco-Manipulation]]: Walk & Pick 任务的核心能力

### 硬件/数据相关

- [[人形机器人|Unitree G1]]: 实验平台
- [[MPJPE]]: 运动追踪精度评估指标

---

## 速查卡片

> [!summary] DreamVLA: From Generated Human Motions to Humanoid Vision-Language-Action Policies
> - **核心**: 生成人体运动引导仿真 RL，Recovery DAgger 蒸馏为 GR00T VLA，无需新遥操作数据
> - **方法**: OmniControl 参考生成 → 残差 SONIC RL 专家 → 参考条件 Recovery DAgger → GR00T 微调
> - **结果**: 3 轮 DAgger 后 Pick 54%、Drawer 60%、Button 69%、Walk&Pick 18%（从 Offline 的 0%/51%/59%/0% 提升）
> - **代码**: 未开源（项目主页：https://ember-center-berkeley.github.io/DreamVLA/）

---

*笔记创建时间: 2026-11-11*
