---
title: "Alaya-EVOKE: From Linear-Scaling Supervision to Endless World"
method_name: "Evoke"
authors: [Yuanyang Yin, Gongxuan Wang, Yifan Zhan, Chuanhao Li, Kaipeng Zhang, Feng Zhao]
year: 2026
venue: arXiv
tags: [world-model, video-generation, geometric-memory, knowledge-distillation, long-horizon-supervision]
zotero_collection: 1-生成模型
image_source: online
arxiv_html: https://arxiv.org/html/2608.13546
created: 2026-08-25
---

# 论文笔记：Alaya-EVOKE: From Linear-Scaling Supervision to Endless World

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | MoE Key Lab of BIPC, USTC；上海创新研究院；Alaya Lab |
| 日期 | 2026年8月 |
| 项目主页 | [evoke-world.github.io](https://evoke-world.github.io/Evoke/) |
| 对比基线 | [[Matrix-Game]]、[[Wan 2.2 A14B]] |
| 链接 | [arXiv](https://arxiv.org/abs/2608.13546) / [Code](https://github.com/SII-YuanyangYin/Evoke) |

---

## 一句话总结

> Evoke 通过将世界几何状态外化为相机索引的记忆库，配合线性增长的长视野监督和三步无 CFG 蒸馏，实现了每步计算开销有界的无限时长交互式视频世界生成。

---

## 核心贡献

1. **外部几何世界状态库 ([[世界状态库|World State Bank]])**: 将场景几何维护在外部相机索引的记忆库中，解耦持久空间状态与去噪器，使去噪器上下文不随生成时长增长而膨胀
2. **线性增长长视野教师**: 分块稀疏注意力（局部 + 远端帧检索 + [[线性注意力]]全局状态）实现 30 秒监督窗口，计算量线性而非二次增长
3. **三步无 CFG 蒸馏**: 在 [[Distribution Matching Distillation|分布匹配]] 目标下的自强制 rollout 蒸馏，去除[[Classifier-Free Guidance (CFG)|CFG]]，每 1.5 秒块仅需 2.11 秒在单张 H200 上生成

---

## 问题背景

### 要解决的问题

交互式[[Video World Model|视频世界模型]]面临三个相互冲突的需求：（1）无限时长生成而不引入线性增长的计算开销；（2）长期一致性而不发生[[视频生成漂移|内容漂移]]；（3）实时响应用户指令和文本条件切换。

### 现有方法的局限

- 基于**像素历史**的方法（如 [[Matrix-Game]]）将所有已生成帧堆入上下文，导致上下文随时间线性膨胀
- 基于**外显几何**的方法受限于低质量几何先验和静态场景假设
- 短视野监督训练的教师无法教导学生抵抗长期漂移，生成质量随时长退化

### 本文的动机

将场景几何**外化**到独立[[世界状态库]]，使去噪器每步输入维度固定；同时用线性增长的长视野监督窗口训练教师，再通过[[Distribution Matching Distillation|分布匹配蒸馏]]将长视野能力传递给快速学生。

---

## 方法详解

### 模型架构

Evoke 采用**循环读写**架构，整合[[Video World Model|视频世界模型]]与外部几何记忆：

- **输入**: 相机位姿 $P_k$ + 文本条件 $c_k$ + 有界历史 $h_k$
- **Backbone**: [[Wan 2.2 A14B]]（14B 参数[[扩散模型]]）
- **核心模块**: [[世界状态库]] 用于外化持久几何，分块[[线性注意力|稀疏注意力]]用于长视野教师
- **输出**: 视频 chunk $x_k$（1.5 秒，9 帧，384×640）

每步 Read → Generate → Write 的循环确保 per-step 计算代价恒定。

### 核心模块

#### 模块 1: 世界状态库 (World State Bank)

**设计动机**: 利用[[相机姿态]]索引将三维场景几何外化，实现有界上下文，支持相机回环重访

**具体实现**:
- 运行[[深度估计|单目深度估计]]获取每帧深度图，反投影到世界坐标系得到点云
- 检索时以当前相机位姿为查询，提取当前视野内可见点的几何特征渲染
- **可见性遮罩**控制信息流向：未被历史帧覆盖的区域不强制匹配，允许自由生成
- 保留约 90 秒的几何历史（实验验证覆盖回环间隔即可有效重访）

#### 模块 2: 长视野教师 (Long-Horizon Teacher)

**设计动机**: 普通因果 Transformer 的二次复杂度在 30 秒（189 latent frames）上不可承受

**具体实现**:
- 基于 [[Wan 2.2 A14B]] 修改，引入**分块[[线性注意力|稀疏注意力]]**
- 每个 query chunk 固定访问四类 key：(a) 本块内帧（局部）；(b) 相邻少量远端帧；(c) 以相机位姿相似度检索的历史帧；(d) [[线性注意力]]维护的全局压缩状态
- 支持**按块切换文本条件**，实现动态指令控制（Evocation 能力）
- 监督窗口约 31.4 秒（189 latent frames），相比短视野教师显著降低长期光度漂移

#### 模块 3: 三步无 CFG 蒸馏 (Few-Step Student)

**设计动机**: 教师推理速度不满足交互需求；[[Classifier-Free Guidance (CFG)|CFG]] 额外增加 2× 计算量

**具体实现**:
- **自强制 rollout**: 21 个 chunk（1 真实 + 20 由上步模型生成）建立长期[[Distribution Matching Distillation|分布匹配]]目标
- **分离历史（Detached History）**: chunk 间梯度截断，防止激活内存爆炸，同时保留 20 chunk 监督视野
- **三步粗-细推理**: 在三个从粗到细的去噪步骤中注入[[世界状态库]]几何条件，无需 CFG
- 训练阶段：Long-distill（6×8 GPU，1981 steps）→ Short post-distill 微调

---

## 关键公式

### 公式 1: [[世界状态库|循环世界生成接口]]

$$
r_k = \text{Read}(M_k, P_k), \quad x_k \sim p_\theta(\cdot \mid r_k, h_k, c_k), \quad M_{k+1} = \text{Write}(M_k, x_k, P_k)
$$

**含义**: Evoke 的循环生成接口：每步从[[世界状态库]]读取几何渲染、生成新 chunk、再写回更新状态，per-step 代价固定。

**符号说明**:
- $M_k$: 第 $k$ 步的世界状态库（相机索引的几何记忆）
- $P_k$: 当前相机位姿序列
- $r_k$: 检索到的视角相关几何渲染
- $h_k$: 历史 chunk 信息（固定有界窗口）
- $c_k$: 文本条件（支持逐块切换）
- $x_k$: 新生成的视频 chunk（1.5 秒）

### 公式 2: [[Distribution Matching Distillation|长视野分布匹配目标]]

$$
\mathcal{L}_W(\theta) = \mathbb{E}_k\!\left[ D\!\left(q_\theta^{(k:k+W-1)} \;\|\; p^{(k:k+W-1)}\right) \right]
$$

**含义**: 教师的长视野监督目标，要求学生在 $W$ 个 chunk 的滚动窗口内与真实分布对齐，而非仅对齐单步。

**符号说明**:
- $W$: 监督窗口长度（约 21 个 chunk，对应约 30 秒）
- $q_\theta^{(k:k+W-1)}$: 学生在 chunk $k$ 到 $k+W-1$ 的生成分布（自强制 rollout）
- $p^{(k:k+W-1)}$: 真实视频分布
- $D(\cdot \| \cdot)$: [[Distribution Matching Distillation|分布匹配]]距离度量

### 公式 3: [[Distribution Matching Distillation|蒸馏生成损失]]

$$
\begin{aligned}
\Delta s &= s_{\text{fake}} - s_{\text{real}} \\
\nu &= \operatorname{mean}_\Omega\!\bigl[|\hat{x}_0 - s_{\text{real}}|\bigr] \\
\mathcal{L}_{\text{gen}} &= \tfrac{1}{2}\left\|\hat{x}_0 - \bigl(\hat{x}_0 - \Delta s/\nu\bigr)^{\!\text{detach}}\right\|_2^2
\end{aligned}
$$

**含义**: 学生训练的生成损失，通过标准化分数差异驱动学生分布向真实分布收敛。

**符号说明**:
- $s_{\text{fake}},\, s_{\text{real}}$: 学生生成样本与真实样本的噪声预测分数
- $\hat{x}_0$: 学生预测的去噪图像
- $\nu$: 归一化因子（逐样本像素绝对误差均值，防止尺度爆炸）
- $\Omega$: 空间域
- $(\cdot)^{\text{detach}}$: 停止梯度操作

---

## 关键图表

### Figure 1: 两小时不间断生成

![Figure 1](https://arxiv.org/html/2608.13546v2/fig_l_hourscale_demo.png)

**说明**: Evoke 在连续相机控制下的两小时完整 rollout，三步无 CFG 推理，展示系统在小时级生成中的视觉一致性与可控性。

### Figure 2: 时序提示切换（Evocation）

![Figure 2](https://arxiv.org/html/2608.13546v2/fig_j2_evocation_cases.png)

**说明**: 相同输入图像、相机轨迹和三步学生，仅文本条件不同，控制天空外观同时保持锚定几何（地面、建筑）不变。文本引导未锚定区域成功率约 67%，锚定几何区域修改成功率仅 4%。

### Figure 3: 有界推理开销与世界状态库检索

![Figure 3](https://arxiv.org/html/2608.13546v2/fig_n_inference.png)

**说明**: 相机位姿检索[[世界状态库]]，将当前视角所需几何渲染注入学生模型最粗粒度的三步推理，per-step 代价不随会话时长积累。

### Figure 4/5/6: 长视野教师结构与稳定性对比

![Figure 4-6](https://arxiv.org/html/2608.13546v2/fig_m_teacher_training.png)

**说明**: (a) 65 分钟 rollout 的光度统计在初始瞬态后趋于稳定，说明有界漂移；(b) 分块[[线性注意力|稀疏注意力]]结构：每个 query chunk 访问固定窗口局部帧、远端检索帧和线性注意力全局状态；(c) 对比长/短视野教师，长视野教师蒸馏的学生光度漂移显著更小。

### Figure 7: 世界状态库支撑相机回环重访

![Figure 7](https://arxiv.org/html/2608.13546v2/fig_k_shortturn_memory.png)

**说明**: 三个保留街道场景沿同一 5.9 秒轨迹生成，相机离开后返回时[[世界状态库]]有效恢复原始场景外观，验证几何保留的回环一致性。

### Figure 11: 教师四分钟 12 提示调度 rollout

![Figure 11](https://arxiv.org/html/2608.13546v2/fig_o_teacher_prompt_schedule.png)

**说明**: Evoke Teacher（非三步学生）在 12 个提示调度下的四分钟 rollout，验证教师本身对长期文本条件动态切换的支持能力。

### Figure 12: 多场景定性展示

![Figure 12](https://arxiv.org/html/2608.13546v2/fig_p_showcase_rollouts.png)

**说明**: 九个不同场景的三步无 CFG rollout 定性结果，涵盖室外街道、室内等多种环境。

---

### Table 1: WBench 导航分项（n=158）

| 方法 | 视频质量↑ | 场景设置↑ | 物理↑ | 备注 |
|------|----------|----------|-------|------|
| **Evoke（本文）** | **82.79** | **83.76** | **72.06** | few-step，无 CFG |
| 其他系统（共 9 个） | — | — | — | 含 HY-World、Matrix-Game 等 |

**说明**: Evoke 在视频质量、场景设置、物理合理性三项均排名第一（few-step 系统）。

### Table 2: 公开排行榜总结

| 基准 | Evoke 得分 | 排名 |
|------|-----------|------|
| VBench-2.0 | 66.77 | **第 1** / 10 |
| VBench-Long | 85.11 | 第 7 / 10 |

**说明**: VBench-2.0 综合评分排名第一，VBench-Long 排名第七（长视频一致性指标仍有提升空间）。

---

## 实验

### 数据集

| 数据集 | 特点 | 用途 |
|--------|------|------|
| Sekai | 高质量户外视频 | 训练 |
| 内部视频数据 | 多场景增强 | 训练 |
| WBench（n=158） | 导航任务交互世界模型评测 | 测试 |
| VBench-2.0 | 公开视频生成质量排行榜 | 测试 |
| VBench-Long | 公开长视频质量排行榜 | 测试 |

### 实现细节

- **Backbone**: [[Wan 2.2 A14B]]（14B 参数扩散 Transformer）
- **教师训练流程**: 相机控制预训练 → Long-distill（6×8 GPU，1981 steps）→ Short post-distill 微调
- **推理分辨率**: 384×640，每 chunk 9 帧，1.5 秒
- **推理速度**: 单张 H200，2.11 秒/chunk（3 步，无 CFG，无加速技巧）
- **世界状态库保留窗口**: 90 秒几何历史
- **长视野监督窗口**: 约 31.4 秒（189 latent frames，21 chunks = 1 真实 + 20 生成）

### 关键消融发现

- 几何保留时长 < 回环间隔时，重访质量退化至无内存基线水平；覆盖间隔后质量显著改善
- 长视野教师（30 秒监督窗口）蒸馏的学生比短视野教师蒸馏的学生光度漂移更小
- 文本引导对未锚定内容（如天空）有效（67% 成功率），对已锚定几何区域几乎无效（4%）

---

## 批判性思考

### 优点
1. **有界计算设计**优雅：外部几何记忆将问题分解为固定代价循环步，可扩展性强
2. **长视野蒸馏**从根本上解决了"学生漂移"问题，比单纯增加去噪步骤更合理
3. 无 CFG 情况下达到 VBench-2.0 第一，说明分布匹配蒸馏质量极高

### 局限性
1. **几何表示粗糙**: 单目深度→点云方案难以保留精细物体结构，动态物体无法建模
2. **静态几何假设**: [[世界状态库]]不包含动态状态（如行人运动），限制可变环境支持
3. **推理速度未达实时**: 2.11 秒/块（1.5 秒视频）尚不足以真正实时交互

### 潜在改进方向
1. 引入对象级表示（[[3D Gaussian Splatting]] / NeRF 节点）丰富几何细节
2. 建模动态世界状态（物体运动、状态变化）
3. 进一步蒸馏或量化加速以达到真实时延要求

### 可复现性评估
- [x] 代码开源（[GitHub](https://github.com/SII-YuanyangYin/Evoke)）
- [ ] 预训练模型（暂未发布）
- [x] 训练细节完整（论文附录）
- [ ] 数据集可获取（内部数据不公开）

---

## 关联笔记

### 基于
- [[Wan 2.2 A14B]]: 教师 Backbone（14B 扩散 Transformer）
- [[Distribution Matching Distillation]]: 蒸馏目标框架

### 对比
- [[Matrix-Game]]: 基于像素历史的世界模型对比基线
- [[Video World Model]]: 核心研究领域归属

### 方法相关
- [[世界状态库]]: 本文核心创新（新概念）
- [[线性注意力]]: 长视野教师的全局状态组件
- [[视频生成漂移]]: 本文要解决的核心问题之一
- [[Classifier-Free Guidance (CFG)]]: 蒸馏中去除的组件
- [[深度估计]]: 构建世界状态库的几何来源

### 硬件/数据相关
- H200 GPU: 推理和训练硬件

---

## 速查卡片

> [!summary] Alaya-EVOKE
> - **核心**: 外部几何记忆库 + 线性增长长视野监督 → 无限时长交互式视频生成
> - **方法**: 循环读写世界状态库、分块稀疏注意力教师、三步无CFG蒸馏
> - **结果**: VBench-2.0 第1（66.77），WBench 质量/场景/物理三项第一，2小时连续生成
> - **代码**: [GitHub](https://github.com/SII-YuanyangYin/Evoke)

---

*笔记创建时间: 2026-08-25*
