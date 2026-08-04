---
type: concept
aliases: [伪逆引导, 掩码伪逆引导, Masked Pseudoinverse Guidance, Moore-Penrose Guidance]
---

# Pseudoinverse Guidance

## 定义

推理时引导技术：在扩散/流匹配模型的每个 solver step 处，利用 Moore-Penrose 伪逆将外部观测约束注入生成速度场，实现无需重训练的条件生成或闭环修正。

## 数学形式

设测量模型为 $\mathbf{Y} = h(\mathbf{X}^1) + \eta$，引导速度场更新为：

$$
v_{\text{guided}} = \bar{v} + \lambda (\mathbf{J})^\top \mathbf{P} \mathbf{W} (\mathbf{Y} - \hat{\mathbf{X}}^1)
$$

其中：
- $\hat{\mathbf{X}}^1 = \mathbf{X}^\tau + (1-\tau)\bar{v}$：当前端点估计
- $\mathbf{J} = \partial \hat{\mathbf{X}}^1 / \partial \mathbf{X}^\tau$：端点关于迭代点的雅可比矩阵
- $\mathbf{W}$：掩码矩阵（选取受约束坐标）
- $\mathbf{P}$：模态预调节器
- $\lambda$：自适应步长增益

对线性测量 $\mathbf{Y} = \mathbf{H}\mathbf{X}^1$，校正量 $\mathbf{H}^\dagger(\mathbf{Y} - \mathbf{H}\hat{\mathbf{X}}^1)$ 精确对应 Moore-Penrose 伪逆修正。

## 核心要点

1. **无需重训练**: 仅在推理时修改速度场，生成模型参数保持冻结
2. **掩码选择性**: 通过 $\mathbf{W}$ 矩阵仅在有观测的坐标施加约束，其余坐标自由生成
3. **雅可比传播**: 利用 VJP（向量-雅可比积）将测量残差高效传播回输入空间
4. **自适应增益**: $\lambda_\tau = \min(\beta, (1-\tau)/(\tau r_\tau^2))$ 在噪声端避免过修正
5. **对齐坐标假设**: 当 $h^\dagger(h(\hat{\mathbf{X}})) \approx \hat{\mathbf{X}}$ 时近似精确（适合潜变量/动作空间）

## 代表工作

- [[FBFM]]: 在 World-Action Model 推理中用掩码伪逆引导实现块内闭环反馈
- ΠGD (Pseudoinverse-Guided Diffusion): 扩散模型的推理时逆问题求解先驱

## 相关概念

- [[Flow-Matching]]
- [[JVP]]
- [[Real-Time Chunking]]
- [[WAM]]
