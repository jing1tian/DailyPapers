---
title: "LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies"
method_name: "LaWAM"
authors: [Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, Yuanbo Xu, Chao Yu]
year: 2026
venue: CoRL 2026
tags: [world-action-model, latent-world-model, robot-manipulation, flow-matching, diffusion-transformer, knowledge-insulation, latent-action-model]
zotero_collection: Robotics/World Model
image_source: online
arxiv_html: https://arxiv.org/html/2606.15768
created: 2026-10-28
---

# 论文笔记：LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies

## 元信息

| 项目 | 内容 |
|------|------|
| 机构 | RLINF Team |
| 日期 | June 2026 (v2: October 2026) |
| 项目主页 | [rlinf.github.io/LaWAM](https://rlinf.github.io/LaWAM/) |
| 对比基线 | [[Faster-WAM]], [[LingBot-Video]], [[Fast-WAM]] |
| 链接 | [arXiv](https://arxiv.org/abs/2606.15768) / Code: — |

---

## 一句话总结

> LaWAM 将 [[Latent-Action|潜在动作模型]] 的解码器复用为紧凑潜在世界模型（LaWM），以单次前向推理生成视觉子目标，以 24× 低于像素空间 WAM 的延迟实现动态感知的机器人策略。

---

## 核心贡献

1. **潜在世界模型（LaWM）复用**: 将预训练 [[ALAM|潜在动作模型（LAM）]] 的前向解码器直接复用为策略的世界模型，无需重新设计世界模型模块，一次非迭代前向推理产生潜在子目标。
2. **知识隔离（Knowledge Insulation）**: 对预测子目标 $\hat{u}_T$ 应用 stop-gradient，阻止动作梯度反向传播到 LaWM 分支，维护 Stage 1 学到的动作相关表示。
3. **Alternate-DiT 动作专家**: 交替注意 VLM 语义 token 和空间动态 token（当前 + 子目标特征），融合高层语义与视觉未来，基于 [[Flow-Matching|条件流匹配]] 生成动作块。
4. **混合频率训练 + 物理时间编码**: 对 5/10/20 Hz 混合数据用正弦物理时间编码对齐动作 token，解决控制频率混乱问题。

---

## 问题背景

### 要解决的问题

[[Vision-Language-Action|VLA]] 策略缺乏对机器人动作如何改变场景的显式预见。[[世界动作模型（WAM）]] 虽然引入了世界模型，但依赖像素空间视频生成，存在大量像素级冗余，导致推理延迟极高（LingBot-VA: 4,482 ms/chunk）。

### 现有方法的局限

- **像素空间 WAM**（LingBot-VA 等）：生成未来视频帧，延迟高，计算开销大（5B+ 参数视频生成器），难以部署。
- **辅助监督方法**（Mantis、VLA-JEPA）：训练时做未来预测监督，但推理时丢弃预测器，未来信息无法条件化动作生成。
- **效率优化 WAM**（Fast-WAM、GigaWorld-Policy）：降低世界模型深度，但削弱了未来感知的实际作用。
- **LDA-1B**：在 DINO 潜在空间建模动态，但使用联合扩散去噪，耦合度高。

### 本文的动机

[[ALAM|潜在动作模型（LAM）]] 的前向解码器本身已是一个潜在动作条件的世界模型：给定当前帧特征 $u$ 和潜在动作 $z$，解码器预测地平线帧特征 $u_T$。LaWAM 直接复用这一解码器作为 LaWM，策略先验预测 $\hat{z}$，LaWM 单次前向解码出子目标 $\hat{u}_T$，再条件化动作生成。

---

## 方法详解

### 模型架构

LaWAM 采用**两阶段训练**架构：

- **输入**: 语言指令 $l$ + 当前 RGB 观测 $o_t$（256×256，无本体感知）
- **视觉骨干**: 冻结 [[DINOv3]]（提取特征 $u, u_T$）
- **Stage 1 骨干（LaWM）**: 两个 24 层 Transformer（V-JEPA 2 风格时空编码），约 230M 参数
- **Stage 2 VLM 骨干**: Qwen3-VL 前 16 层
- **动作专家**: 4 块 Alternate-DiT，隐藏维度 1024
- **输出**: [[Action Chunking|动作块]] $a_{1:T}$
- **总参数**: 2.3B（策略），LaWM 约 230M

### 核心模块

#### 模块 1：Latent Action Model / Latent World Model（Stage 1）

**设计动机**: 利用 [[ALAM|潜在动作模型]] 的架构天然实现动态先验学习，解码器即世界模型。

**具体实现**:

1. **后验编码器（逆向动力学）**: 从当前帧和地平线帧的 [[DINOv3]] 特征 $(u, u_T)$ 推断潜在动作 $z \sim q_\phi(z \mid u, u_T)$（KL 正则到标准正态）
2. **前向解码器（LaWM）**: $\tilde{u}_T = \text{LaWM}_\omega(u, z)$，通过自适应 LayerNorm 将 $z$ 注入 Transformer 解码器
3. **辅助解码器（训练后丢弃）**: 从机器人状态 $s$ 和 $z$ 预测地平线末端执行器状态 $s_T$，引导潜在动作空间学习动作相关结构
4. 训练数据：约 3,000 小时机器人视频 + 1,500 小时以自我为中心的人类视频（仅用于 LaWM 动态先验）

#### 模块 2：策略先验 + Knowledge Insulation（Stage 2）

**设计动机**: 在 Stage 1 冻结 LaWM 的基础上，训练策略先验直接从 $(o, l)$ 回归 $\hat{z}$，再经 LaWM 解码子目标，条件化动作专家。

**具体实现**:

1. **策略先验**: 基于 Qwen3-VL 前 16 层，从语言指令和当前观测预测潜在动作 $\hat{z}$
2. **LaWM 解码**: 单次前向推理得到预测子目标 $\hat{u}_T = \text{LaWM}_\omega(u, \hat{z})$
3. **[[知识隔离（Knowledge Insulation）|Knowledge Insulation（KI）]]**: 对 $\hat{u}_T$ 应用 stop-gradient 后送入动作专家，阻止动作损失梯度流回 LaWM，保护 Stage 1 学到的表示
4. **Alternate-DiT 动作专家**: 4 块 DiT，交替 cross-attend VLM 语义 token（高层语义上下文）和空间动态 token（$u$ 与 $\operatorname{sg}[\hat{u}_T]$ 拼接），最终通过 [[Flow-Matching|条件流匹配]] 回归动作块

#### 模块 3：混合频率训练 + 物理时间编码

**设计动机**: 机器人数据集控制频率不一（5/10/20 Hz），同一时间间隔下动作 token 的时序对齐依赖物理时间而非帧索引。

**具体实现**:

- 对每个控制频率 $h$，动作块长度 $T = \text{round}(\tau h)$，时间步 $t_i = i/h$
- 对 $t_i$ 应用正弦编码 $\text{PE}(t_i)$ 注入到动作 token，使不同频率的动作 token 在物理时间维度对齐

---

## 关键公式

### 公式 1：[[世界动作模型（WAM）|问题建模：联合分解]]

$$
p(a_{1:T}, o_T \mid o, l) = p(o_T \mid o, l)\, p(a_{1:T} \mid o, l, o_T)
$$

**含义**: 将世界动作模型建模为先预测未来观测 $o_T$，再在未来观测条件下生成动作块 $a_{1:T}$，相比像素空间 WAM 的联合分布更高效。

**符号说明**:
- $o$: 当前观测
- $l$: 语言指令
- $o_T$: 地平线帧（chunk 末尾观测）
- $a_{1:T}$: 动作块

### 公式 2：[[ALAM|LaWM 后验与解码]]

$$
z \sim q_\phi(z \mid u, u_T), \qquad \tilde{u}_T = \mathrm{LaWM}_\omega(u, z)
$$

**含义**: 后验从当前和地平线特征推断潜在动作 $z$；LaWM 解码器以 $u$ 和 $z$ 为输入，重建地平线特征 $\tilde{u}_T$。

**符号说明**:
- $u = f_\text{DINOv3}(o)$: 当前帧视觉特征
- $u_T = f_\text{DINOv3}(o_T)$: 地平线帧视觉特征
- $z$: 潜在动作（KL 正则到 $\mathcal{N}(0, I)$）
- $q_\phi$: 后验编码器（逆向动力学）
- $\omega$: LaWM 解码器参数

### 公式 3：[[知识隔离（Knowledge Insulation）|LaWAM 推理分解]]

$$
p(a_{1:T}, \hat{u}_T, \hat{z} \mid o, l) = p_\theta(\hat{z} \mid o, l)\, p_\omega(\hat{u}_T \mid u, \hat{z})\, p_\eta(a_{1:T} \mid o, l, u, \hat{u}_T)
$$

**含义**: Stage 2 推理分解为三个连续步骤：策略先验预测 $\hat{z}$，LaWM 解码 $\hat{u}_T$，动作专家在 $(o, l, u, \hat{u}_T)$ 条件下生成动作块。

**符号说明**:
- $\theta$: 策略先验参数
- $\omega$: LaWM（Stage 1 冻结）参数
- $\eta$: 动作专家参数
- $\hat{u}_T$: 预测的潜在子目标

### 公式 4：[[ALAM|Stage 1 LaWM 训练损失]]

$$
\mathcal{L}_{\mathrm{LAM}} = \mathcal{L}_{\mathrm{wm}}^{(1)} + \mathcal{L}_{\mathrm{aux}} + \beta\, \mathbb{E}\!\left[D_{\mathrm{KL}}\!\left(q_\phi(z \mid u, u_T) \,\|\, \mathcal{N}(0, I)\right)\right]
$$

$$
\mathcal{L}_{\mathrm{wm}}^{(1)} = \mathbb{E}\!\left[\lVert \tilde{u}_T - u_T \rVert_2^2\right], \qquad \mathcal{L}_{\mathrm{aux}} = \mathbb{E}\!\left[\lVert g(s, z) - s_T \rVert_2^2\right]
$$

**含义**: Stage 1 三项损失：前向预测 MSE 监督解码器预测地平线特征，辅助 MSE 监督潜在动作编码运动信息（训练后 $g$ 丢弃），KL 正则化压缩潜在动作空间。

**符号说明**:
- $\mathcal{L}_{\mathrm{wm}}^{(1)}$: 地平线特征前向预测 MSE
- $\mathcal{L}_{\mathrm{aux}}$: 末端执行器状态辅助预测 MSE
- $g(s, z)$: 辅助解码器，从状态 $s$ 和潜在动作 $z$ 预测 $s_T$
- $\beta = 10^{-5}$: KL 权重

### 公式 5：[[知识隔离（Knowledge Insulation）|Stage 2 LaWAM 联合训练损失]]

$$
\mathcal{L}_{\mathrm{LaWAM}} = \lambda_{\mathrm{distill}}\, \mathcal{L}_{\mathrm{distill}} + \lambda_{\mathrm{wm}}\, \mathcal{L}_{\mathrm{wm}}^{(2)} + \mathcal{L}_{\mathrm{act}}
$$

$$
\mathcal{L}_{\mathrm{distill}} = \mathbb{E}\!\left[\lVert \hat{z} - z \rVert_2^2\right], \qquad \mathcal{L}_{\mathrm{wm}}^{(2)} = \mathbb{E}\!\left[\lVert \hat{u}_T - u_T \rVert_2^2\right]
$$

**含义**: 三项联合损失：潜在动作蒸馏监督策略先验预测 $\hat{z}$（来自 Stage 1 后验 $z$），未来预测损失监督 LaWM 在策略预测 $\hat{z}$ 下仍能重建地平线特征，动作损失训练动作专家。Knowledge Insulation 的 $\operatorname{sg}[\hat{u}_T]$ 阻止动作梯度流回预测子目标分支。

**符号说明**:
- $\lambda_{\mathrm{distill}} = \lambda_{\mathrm{wm}} = 0.1$: 权重系数
- $\mathcal{L}_{\mathrm{distill}}$: 潜在动作蒸馏损失（策略先验对 Stage 1 后验的蒸馏）
- $\mathcal{L}_{\mathrm{wm}}^{(2)}$: Stage 2 前向预测损失
- $\mathcal{L}_{\mathrm{act}}$: [[Flow-Matching|条件流匹配]] 动作损失

### 公式 6：[[混合频率训练|物理时间对齐]]

$$
T = \mathrm{round}(\tau h), \qquad t_i = \frac{i}{h}, \quad i = 1, \ldots, T
$$

**含义**: 给定物理时间间隔 $\tau$（固定）和控制频率 $h$，计算动作块长度 $T$ 及每个动作步的物理时间戳 $t_i$。

**符号说明**:
- $\tau$: 固定物理时间间隔（chunk 对应的秒数）
- $h$: 数据集控制频率（如 5/10/20 Hz）
- $T$: 动作块长度

### 公式 7：[[混合频率训练|正弦物理时间编码]]

$$
\mathrm{PE}(t_i) = \mathrm{Concat}\!\left[\sin(t_i \nu_k), \cos(t_i \nu_k)\right]_{k=0}^{K-1}
$$

$$
\nu_k = \exp\!\left(-\log(P_{\max}) \cdot \frac{k}{\max(K-1, 1)}\right), \qquad K = \lceil d/2 \rceil
$$

**含义**: 对物理时间步 $t_i$ 施加多频率正弦编码，注入动作 query 嵌入，使不同控制频率的动作 token 在物理时间维度对齐，解决混合频率训练中的时序混乱。

**符号说明**:
- $\nu_k$: 第 $k$ 个正弦频率
- $P_{\max}$: 最大周期（频率范围参数）
- $d$: 动作 query 嵌入维度
- $K = \lceil d/2 \rceil$: 频率数量

---

## 关键图表

### Figure 1：LIBERO 延迟-成功率权衡

![Figure 1](https://rlinf.github.io/LaWAM/assets/figures/libero_latency.png)

**说明**: 在 A100 GPU 上（10 步去噪）绘制各方法的 LIBERO 成功率 vs 推理延迟。LaWAM（187 ms，98.6%）在成功率和延迟上均优于 Fast-WAM（486 ms，97.6%）和 LingBot-VA（4,482 ms，98.5%）。标记面积代表模型参数量，粉色扇形为世界模型参数。

### Figure 2：LaWAM 两阶段管线概览

![Figure 2](https://arxiv.org/html/2606.15768v2/LaWAM_overview.png)

**说明**: Stage 1 训练 [[ALAM|潜在动作模型（LAM）]] + [[DINOv3]] 特征空间的世界模型（LaWM）；Stage 2 冻结 LaWM，训练策略先验 + [[知识隔离（Knowledge Insulation）|Knowledge Insulation]] + Alternate-DiT 动作专家。

### Figure 3：子目标引导的 Chunk 执行

![Figure 3](https://arxiv.org/html/2606.15768v2/libero_chunk_1.png)

**说明**: 上行展示 LIBERO 中一个执行 chunk 内的观测序列与预测潜在子目标；下行叠加子目标导向的机械臂热图，显示实际运动轨迹趋向预测子目标区域，验证子目标条件化的有效性。

### Figure 4：真实场景双平台操作

![Figure 4](https://rlinf.github.io/LaWAM/assets/figures/real_world_rollouts.png?v=20261010)

**说明**: 在两种机器人平台上的真实操作：(a) Franka Emika Panda 执行 Pick-and-Place（93.3%）和 Open Drawer（86.7%）；(b) Quanta X1 双臂机器人执行 Fold Towel（90.0%）和八阶段 Clean Table（42.0%）。

### Figure 5：跨平台开环 LaWM 滚动

![Figure 5](https://arxiv.org/html/2606.15768v2/lam_intro_pca.png)

**说明**: 将同一潜在动作轨迹施加到不同未见场景和机器人平台的初始观测，LaWM 产生上下文相关的潜在变化，验证 LaWM 的跨平台泛化能力。

### Figure 6：LaWM 滚动定量结果

![Figure 6](https://arxiv.org/html/2606.15768v2/picture/error_curves.png)

**说明**: 在 500 条 LIBERO 轨迹上统计特征余弦相似度：LaWM 滚动 vs 真实地平线特征（保持高相似度），以及滚动 vs 初始帧（随时间步持续偏离），验证 LaWM 成功编码动作诱导的场景变化。

### Figure 7：单臂跨平台滚动

![Figure 7](https://arxiv.org/html/2606.15768v2/lam_single_pca.png)

**说明**: 单臂机器人场景的跨平台开环滚动可视化（PCA 降维展示特征空间轨迹）。

### Figure 8：双臂跨平台滚动

![Figure 8](https://arxiv.org/html/2606.15768v2/lam_dual_pca.png)

**说明**: 双臂机器人（含未见 π 平台截图）的跨平台开环滚动可视化，包含 pi.website 场景。

### Figure 9：预测接口设计比较

![Figure 9](https://arxiv.org/html/2606.15768v2/picture/lawam_three_styles_v2.png)

**说明**: 三种预测接口设计对比：(a) 潜在动作监督（LAPA、UniVLA）——训练时监督，推理时无条件化；(b) 未来预测监督（Mantis、VLA-JEPA）——训练辅助监督，推理时丢弃未来预测器；(c) **LaWAM**——推理时保留 LaWM，动作生成以预测空间子目标为条件（实线为推理路径，虚线为仅训练监督）。

### Figure 10：混合频率训练结果

**说明**（图片见附录）: 在 LIBERO 数据降采样至 5/10/20 Hz 的联合训练实验中，无物理时间编码时性能下降显著，添加正弦物理时间编码后大幅恢复，验证 [[混合频率训练|物理时间对齐]] 的必要性。

### Figure 11：LIBERO 定性示例（附录）

**说明**: LaWAM 在 LIBERO 上的动作执行序列，附带潜在世界滚动可视化，展示紧凑潜在子目标如何引导动作生成。

### Figure 12：真实场景机器人设置（附录）

**说明**: (a) Franka Emika Panda + 外置 RGB 相机（单臂任务）；(b) Quanta X1 双臂机器人（毛巾折叠和桌面清理任务）。

### Figure 13-16：LaWM 子目标可视化（附录）

**说明**:
- **Fig 13 (Pick-and-Place)**: Franka 臂抓取玩具鸭并放入碗中的潜在子目标预测序列。
- **Fig 14 (Drawer Opening)**: Franka 抓取抽屉把手并拉开的潜在子目标预测序列。
- **Fig 15 (Towel Folding)**: Quanta X1 展开毛巾、压平后完成两次长边折叠和一次短边折叠的预测序列。
- **Fig 16 (Clean Table)**: 在 Quanta X1 上对训练演示的离线回放，采样 28 个 chunk 的潜在子目标序列。

### Table 1：LIBERO 基准对比（成功率 %）

| Method | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-Long | Average |
|--------|---------------|---------------|-------------|-------------|---------|
| π0.5 | — | — | — | — | 98.5 |
| Cosmos-Policy | — | — | — | — | 98.5 |
| LingBot-VA | — | — | — | — | 98.5 |
| Fast-WAM | — | — | — | — | 97.6 |
| **LaWAM** | **99.4** | **99.6** | **98.4** | **97.0** | **98.6** |

**说明**: LaWAM 以 98.6% 平均成功率达到 SOTA 或可比结果，其中 LIBERO-Long（最长时序任务）以 97.0% 超越其他竞争方法。

### Table 2：RoboTwin 2.0 对比（成功率 %，50 任务）

| Method | Clean Avg | Random Avg | Overall |
|--------|-----------|------------|---------|
| π0.5 | — | — | — |
| Fast-WAM | 91.98 | 90.52 | 91.25 |
| GigaWorld-Policy | — | — | — |
| **LaWAM** | **92.64** | **89.80** | **91.22** |

**说明**: LaWAM 在 Clean 场景下以 92.64% 优于 Fast-WAM（91.98%），整体成绩相当，在 100 次试验 × 50 任务的大规模评测中表现稳健。

### Table 3：真实场景成功率（%）

| Task | LaWAM | π0.5 | Fast-WAM | LingBot-VA | GR00T-N1.6 |
|------|-------|-------|----------|------------|------------|
| Pick-and-Place | **93.3** | 86.7 | 56.7 | 76.7 | 83.3 |
| Open Drawer | **86.7** | 80.0 | 63.3 | **83.3** | 76.7 |
| Fold Towel | **90.0** | 83.3 | 70.0 | 0.0 | 46.7 |
| Fold Towel (Light) | **68** | 46 | 14 | — | — |
| Clean Table | **42** | 28 | 6 | — | — |
| Clean Table (Light) | 24 | **28** | 6 | — | — |

**说明**: LaWAM 在大多数真实场景任务中大幅领先，尤其是长时序双臂任务（Fold Towel: 90% vs LingBot-VA: 0%）。LingBot-VA 毛巾折叠失败归因于高推理延迟（4,482 ms）。光照扰动下（Light）LaWAM 仍大幅优于 Fast-WAM，但 Clean Table-Light 略低于 π0.5。

### Table 4：推理效率对比

| Method | Latency (ms) | Parameters | World Model Params |
|--------|-------------|------------|-------------------|
| LingBot-VA | 4,482 | — | ~5B (WAN) |
| Fast-WAM | 486 | — | — |
| π0.5 | 220 | — | — |
| **LaWAM** | **187** | **2.3B** | **230M** |

**说明**: LaWAM 的 LaWM（230M）相比像素空间 WAM 骨干（5B WAN）参数少 95%，推理延迟仅 187 ms，比 LingBot-VA 快 24×，比 Fast-WAM 快 2.6×，比 π0.5 还略快。

### Table 5：组件消融（LIBERO-10，成功率 %）

| 配置 | Success Rate |
|------|-------------|
| 基础 VLA（无世界模型） | 91.0 |
| w/o Stage-2 预训练 | 96.4 |
| w/o Knowledge Insulation | 94.2 |
| w/o 潜在动作蒸馏 | 93.4 |
| w/o KI & 蒸馏 | 93.2 |
| w/o 子目标条件化（仅训练监督） | 94.6 |
| 仅未来预测损失 | 93.4 |
| 仅潜在动作蒸馏 | 95.2 |
| **Full LaWAM** | **97.0** |

**关键发现**: 
- 子目标条件化推理（vs 仅训练监督）提升 2.4%，证明推理时世界模型的价值
- Knowledge Insulation 贡献 2.8%，是最重要的单一组件
- 潜在动作蒸馏贡献 1.8%，两者联用效果超过各自之和

### Table 6：机制验证消融

| 配置 | LIBERO-10 SR% |
|------|--------------|
| 直接预测未来特征（无潜在动作瓶颈） | 94.4 |
| 容量匹配 adapter（随机初始化解码器） | 91.4 |
| **Full LaWAM（复用 LAM 解码器）** | **97.0** |

**关键发现**: 复用预训练 LAM 解码器（vs 随机初始化等容量 adapter）提升 5.6%，证明 Stage 1 预训练的关键价值；潜在动作瓶颈（vs 直接预测特征）提升 2.6%，验证潜在动作空间的优越表示能力。

---

## 实验

### 数据集与训练数据

| 数据集 | 规模 | 特点 | 用途 |
|--------|------|------|------|
| 机器人视频 | ~3,000 小时 | 多任务机器人操作 | Stage 1 LaWM 预训练 |
| 人类以自我为中心视频 | ~1,500 小时 | 双手操作，丰富动态先验 | Stage 1 LaWM 动态先验（仅） |
| LIBERO | 50 任务 × 50 示范 | 4 suites，25k 步训练 | 后训练评测 |
| RoboTwin 2.0 | 50+ 任务，27,500 条示范 | Clean + Randomized 设置 | 后训练评测 |
| 真实机器人数据 | 150-302 条/任务 | Franka + Quanta X1 | 真实场景评测 |

### 实现细节

- **视觉特征**: 冻结 [[DINOv3]]（256×256 RGB，无本体感知）
- **VLM 骨干**: Qwen3-VL 前 16 层
- **动作专家**: 4 块 Alternate-DiT，隐藏维度 1024
- **优化器**: Adam，LaWM 学习率 $3 \times 10^{-5}$，动作专家 $10^{-4}$（后训练 cosine decay）
- **训练成本 (H100 GPU 小时)**: Stage 1 LaWM 预训练 400h，Stage 2 策略预训练 3,000h，LIBERO 后训练 40h，RoboTwin 后训练 1,200h
- **峰值显存**: LaWAM 6.04 GB（vs Fast-WAM 23.68 GB）
- **去噪步数**: 10 步（评测延迟基准）
- **硬件（评测）**: 单卡 A100（BF16），1,000 次重复平均

### 可视化结果

LaWM 开环滚动（Fig 5-8）显示：在未见场景和跨平台（Franka、Quanta X1、π 平台）条件下，相同潜在动作轨迹产生上下文相关的特征变化，证明 LaWM 学到了可迁移的动态先验。LIBERO 的潜在子目标热图（Fig 3）定性验证了动作向预测子目标区域收敛。

---

## 批判性思考

### 优点

1. **高效世界模型**: 复用 LAM 解码器作为 LaWM，230M 参数实现 95% 的参数削减（vs 5B 像素 WAM），推理延迟 187 ms，兼顾效果和效率
2. **设计精简**: Knowledge Insulation 以极低代价（stop-gradient）实现梯度解耦，无需复杂架构改动
3. **数据多样性**: 人类以自我为中心视频提供更丰富的手部操作动态先验，增强 LaWM 泛化性
4. **端到端验证**: 涵盖仿真（LIBERO、RoboTwin）和真实场景（双平台、6 类任务），包括光照扰动鲁棒性测试

### 局限性

1. **大幅度相机运动**: [[DINOv3]] 特征对大视角变化敏感，限制在人形机器人或移动底座平台上的应用
2. **精细形变建模不足**: 微妙布料形变（如 Clean Table 折叠细节）在训练数据中欠采样，LaWM 特征分辨率偏粗
3. **RGB-only 输入**: 无本体感知信息，在遮挡或精确力控场景可能受限
4. **Clean Table-Light 表现**: 光照扰动下 Clean Table 成功率（24%）低于 π0.5（28%），表明颜色增强策略有限

### 潜在改进方向

1. 引入本体感知信息（关节角、末端执行器位姿）到动作专家，增强精细操作能力
2. 提升 LaWM 特征分辨率（如 hierarchical patch），改善精细形变建模
3. 探索 LaWM 在移动操作或人形机器人场景的适配，解决大幅相机运动问题
4. 将 LaWM 扩展至多步预测（多 chunk 预测），支持更长时域规划

### 可复现性评估

- [ ] 代码开源（未在论文中提及）
- [ ] 预训练模型（未在论文中提及）
- [x] 训练细节完整（Appendix 提供完整超参和训练协议）
- [ ] 数据集可获取（机器人和人类视频数据集未公开）

---

## 关联笔记

### 基于

- [[ALAM|Latent Action Model（LAM）]]: LaWAM 直接复用其解码器作为 LaWM
- [[DINOv3]]: 冻结视觉骨干，提供特征空间
- [[VJEPA-2|V-JEPA 2]]: LaWM 编码器的时空设计参照

### 对比

- [[Faster-WAM]]: 像素空间高效 WAM，同为推理效率优化，延迟 486 ms
- [[LingBot-Video]]: 像素空间 WAM，高成功率但延迟 4,482 ms
- [[Fast-WAM]]: 轻量化世界模型，辅助监督策略

### 方法相关

- [[世界动作模型（WAM）]]: LaWAM 属于 WAM 范畴
- [[Flow-Matching|条件流匹配]]: 动作专家采用的生成框架
- [[知识隔离（Knowledge Insulation）]]: LaWAM 核心设计之一
- [[Action Chunking]]: 输出格式
- [[混合频率训练]]: 物理时间编码技术

### 硬件/数据相关

- Franka Emika Panda: 单臂任务实验平台
- Quanta X1: 双臂机器人，用于长时序任务

---

## 速查卡片

> [!summary] LaWAM (CoRL 2026)
> - **核心**: 复用 LAM 解码器作潜在世界模型，单次推理生成子目标条件化动作
> - **方法**: Stage 1 训 LaWM，Stage 2 策略先验 + KI + Alternate-DiT，物理时间编码混频训练
> - **结果**: LIBERO 98.6%、RoboTwin 91.22%、真实场景领先，187 ms/chunk（24× 快于 LingBot-VA）
> - **代码**: 未公开（截至 2026-10）

---

*笔记创建时间: 2026-10-28*
