---
type: concept
aliases: [InstructPix2Pix, IP2P]
---

# InstructPix2Pix

## 定义
Brooks et al. (CVPR 2023) 提出的基于扩散模型的图像编辑方法，能够根据自然语言指令对图像内容进行修改，同时保留无关区域的外观。

## 数学形式
在 Stable Diffusion 基础上，额外以原始图像为条件，使用分类器自由引导（CFG）：
$$\hat{\epsilon}_\theta = \epsilon_\theta(\mathbf{z}_t, \mathbf{c}_T, \mathbf{c}_I) + s_T(\epsilon_\theta(\mathbf{z}_t, \mathbf{c}_T, \mathbf{c}_I) - \epsilon_\theta(\mathbf{z}_t, \varnothing, \mathbf{c}_I)) + s_I(\epsilon_\theta(\mathbf{z}_t, \varnothing, \mathbf{c}_I) - \epsilon_\theta(\mathbf{z}_t, \varnothing, \varnothing))$$
其中 $s_T, s_I$ 分别为文本和图像引导强度。

## 核心要点
1. **双条件引导**: 同时条件化文本指令和源图像，可独立调节两个条件的引导强度
2. **数据构建**: 用 GPT-3 生成指令，再用 Prompt2Prompt 生成成对图像作为训练数据
3. **无需 mask**: 无需手动指定编辑区域，模型自动推断修改位置
4. **早期基础工作**: 后续大量图像编辑模型（MagicBrush、EMU Edit、FLUX-Kontext）均在此基础上改进

## 代表工作
- [[SWEET]]: SWEET 方法比较中引用的早期图像编辑代表工作

## 相关概念
- [[FLUX-Kontext]]
- [[图像编辑世界模型]]
- [[1-生成模型/潜扩散模型.md|潜扩散模型]]
