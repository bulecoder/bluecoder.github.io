# Transformer 核心演进全解（2017-2025 主流方案）
从2017年原始Transformer到现在的主流大模型架构，核心演进完全围绕**「效果提升、显存优化、推理加速、长文本支持」**四大目标，分四大模块讲透，每个模块讲清「原始结构的痛点→演进方案→解决的问题→主流应用」，完全匹配Llama、Qwen等主流基座的架构设计。  
四个重要核心：
- KV Cache应用与推理阶段，因为训练阶段是Teacher Forcing，并行计算所有位置，无需缓存，推理时自回归生成，为了避免重复计算而进行缓存。
- KV Cache存在于自回归解码器以及编码器-解码器部分，纯编码器是一次性双向处理不需要重复使用。
- 加速 Q@K@V 的矩阵相乘，所有历史token的K和V一旦生成就不会改变，缓存以避免重复计算，Q 每一步是当前token的Q是新的，历史token的Q不再需要重复计算不用缓存。
- KV Cache会增加内存占用，用显存换速度策略。

## 模块一：注意力机制演进（MHA → KV Cache → MQA → GQA → MLA）
MHA、交叉注意力机制、稀疏注意力机制、层注意力、动态路由注意力、NAS优化注意力  
注意力机制是Transformer的核心，演进的核心目标是**降低推理显存占用、提升推理速度，同时最小化效果损失**。

### 1. 原始MHA（多头注意力，Multi-Head Attention）  
2017年，Transformer模型提出，论文标题《Attention Is All You Need》
原始Transformer的基础方案，核心逻辑：
- 把输入的特征分成`n`个独立的头，每个头独立计算Q（查询）、K（键）、V（值），每个头负责捕捉不同维度的语义信息；
- 每个头都有自己独立的K/V投影矩阵，n个头就有n组独立的K/V。

**核心痛点**：
- 自回归推理时，KV Cache显存占用极高。比如Llama 2 70B，8k上下文，单批次KV Cache就需要几十GB显存，是推理速度和长文本支持的最大瓶颈；
- 每个头都要存独立的K/V，参数量大，计算量高。

### 2. KV Cache（KV缓存）
**解决的核心问题**：自回归推理时的重复计算，大幅提升推理速度。
#### 原理：
大模型生成文本是一个字一个字吐的，每生成一个新字，都要把之前所有的字重新算一遍注意力，重复计算量极大。
KV Cache的逻辑是：
- 第一次计算时，把之前所有token的K和V都缓存下来；
- 生成新token时，只需要算新token的Q、K、V，再和缓存的历史K/V拼接计算，不用重复算历史token的K/V。

**效果**：推理速度提升10倍以上，是现在所有大模型推理的标配技术。

### 3. MQA（多查询注意力，Multi-Query Attention）
2019年，论文《Fast Transformer Decoding: One Write-Head is All You Need》
**解决的核心问题**：KV Cache显存占用过高的问题。
#### 原理：
- 保留多头的Q（每个头有独立的Q投影矩阵）；
- 所有头**共享同一组K/V投影矩阵**，也就是所有头用同一个K和V，不再每个头单独存。

**优缺点**：
- 优点：KV Cache显存占用直接降低到原来的1/n（n是头数），推理速度大幅提升；
- 缺点：共享K/V会损失少量模型效果，Google的PaLM、GPT-3.5最早采用。

### 4. GQA（分组查询注意力，Grouped-Query Attention）
2023年，LLaMA2,PaLM等，论文标题《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》
**解决的核心问题**：平衡MHA的效果和MQA的速度，是现在的工业界事实标准。
#### 原理：
- 把所有注意力头分成G个组；
- 每个组内共享同一组K/V投影矩阵，组与组之间的K/V独立。

比如Llama 3 8B用了8个KV头，32个Q头，也就是4个Q头共享一组K/V。

**核心优势**：
- 效果几乎和MHA持平，同时显存占用和推理速度接近MQA，完美平衡了效果和性能；
- 现在Llama 2/3、Qwen、DeepSeek、Mistral等几乎所有主流开源大模型，全部采用GQA作为默认注意力方案。

### 5. MLA (多头潜在注意力机制，Multi-head Latent Attention)（[学习视频地址1](https://www.bilibili.com/video/BV1yspRzPEw8/?spm_id_from=333.337.search-card.all.click&vd_source=d3285a2ba86bc368a3901aac90d388ea)）（[学习视频地址2](https://www.bilibili.com/video/BV1AiKPe4Eok/?spm_id_from=333.788.videopod.sections&vd_source=d3285a2ba86bc368a3901aac90d388ea)）
- **提出时间/模型**：2024年，DeepSeek-V2 / DeepSeek-V3
- **论文**：《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model》
- **核心目标**：解决推理时 KV Cache 过大的问题，将内存带宽瓶颈转化为计算瓶颈，大幅提升吞吐量。

---

#### 核心思想：低秩压缩 + 共享潜向量
- 受 LoRA 启发：不缓存完整的 K、V 矩阵，而是缓存一个**低维的共享潜向量** $c^{KV}$。
- 推理时，从 $c^{KV}$ 动态重构出 K 和 V（有点了类似 Bottleneck）。
- **关键创新**：K 和 V 共享同一个压缩表示，进一步压缩缓存大小。

---

#### 一、符号定义
- $d$：嵌入维度 (embedding dimension)
- $n_h$：注意力头数 (number of attention heads)
- $d_h$：每个头的维度 (dimension per head)
- $\mathbf{h}_t \in \mathbb{R}^d$：第 $t$ 个 token 在某一注意力层的输入

---

#### 二、Key 和 Value 的低秩压缩（核心创新）
**步骤 1：下投影压缩**（生成共享的 KV 潜向量）

$$ \begin{bmatrix} \mathbf{k}_t^C \\ \mathbf{v}_t^C \end{bmatrix} = W^{DKV} \mathbf{h}_t, \tag{1} $$

- $W^{DKV} \in \mathbb{R}^{d_c \times d}$：下投影矩阵
- $d_c$：KV 压缩维度，满足 $d_c \ll d_h n_h$
- $\mathbf{c}_t^{KV} = \begin{bmatrix} \mathbf{k}_t^C \\ \mathbf{v}_t^C \end{bmatrix} \in \mathbb{R}^{d_c}$：压缩后的潜向量（**推理时缓存**）

**步骤 2：上投影重构 Key 的内容部分**

$$ [\mathbf{k}_{t,1}^C; \mathbf{k}_{t,2}^C; \dots; \mathbf{k}_{t,n_h}^C] = \mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV}, \tag{2} $$

- $W^{UK} \in \mathbb{R}^{d_h n_h \times d_c}$：Key 的上投影矩阵
- $\mathbf{k}_{t,i}^C$：第 $i$ 个头的 Key 内容部分（不含位置编码）

**步骤 3：生成带 RoPE 的解耦 Key**（位置路径）

$$ \mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t), \tag{3} $$

- $W^{KR} \in \mathbb{R}^{d_h n_h \times d}$：生成解耦 Key 的矩阵
- $\text{RoPE}(\cdot)$：应用旋转位置编码
- $\mathbf{k}_t^R$：所有头共享的 RoPE 部分（**推理时缓存**）

**步骤 4：拼接内容部分和位置部分得到完整的 Key**

$$ \mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R], \tag{4} $$

- 即每个头的 Key 由两部分拼接而成：内容部分 $\mathbf{k}_{t,i}^C$（来自低秩重构）和位置部分 $\mathbf{k}_t^R$（带 RoPE）

**步骤 5：上投影重构 Value**

$$ [\mathbf{v}_{t,1}^C; \mathbf{v}_{t,2}^C; \dots; \mathbf{v}_{t,n_h}^C] = \mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV}, \tag{5} $$

- $W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$：Value 的上投影矩阵
- $\mathbf{v}_{t,i}^C$：第 $i$ 个头的 Value（不含 RoPE）

---

#### 三、Query 的低秩压缩（主要用于减少训练激活内存）

**步骤 1：下投影压缩 Query**

$$ \mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t, \tag{6} $$

- $W^{DQ} \in \mathbb{R}^{d_c' \times d}$：Query 下投影矩阵
- $d_c'$：Query 压缩维度，满足 $d_c' \ll d_h n_h$
- $\mathbf{c}_t^Q$：Query 的压缩潜向量（推理时不缓存）

**步骤 2：上投影重构 Query 的内容部分**

$$ [\mathbf{q}_{t,1}^C; \mathbf{q}_{t,2}^C; \dots; \mathbf{q}_{t,n_h}^C] = \mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q, \tag{7} $$

- $W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}$：Query 上投影矩阵

**步骤 3：生成带 RoPE 的解耦 Query**

$$ [\mathbf{q}_{t,1}^R; \mathbf{q}_{t,2}^R; \dots; \mathbf{q}_{t,n_h}^R] = \mathbf{q}_t^R = \text{RoPE}(W^{QR} \mathbf{c}_t^Q), \tag{8} $$

- $W^{QR} \in \mathbb{R}^{d_h n_h \times d_c'}$：生成解耦 Query 的矩阵

**步骤 4：拼接得到完整的 Query**

$$ \mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R], \tag{9} $$

- 每个头的 Query 由内容部分和带 RoPE 的位置部分拼接而成

---

#### 四、注意力计算

对第 $i$ 个头，输出为：

$$ \mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j \left( \frac{\mathbf{q}_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}} \right) \mathbf{v}_{j,i}^C, \tag{10} $$

- $d_h^R$：解耦 RoPE 向量的维度（隐含在 $\mathbf{k}_t^R$ 中）
- 注意：Value 只使用内容部分 $\mathbf{v}_{j,i}^C$，不带 RoPE

最终输出经过输出投影：

$$ \mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \dots; \mathbf{o}_{t,n_h}], \tag{11} $$

- $W^O \in \mathbb{R}^{d \times d_h n_h}$：输出投影矩阵

---

#### 五、推理时缓存的内容

根据论文，**只有蓝色框中的向量需要缓存**，即：

1. $\mathbf{c}_t^{KV}$（KV 压缩潜向量，维度 $d_c$）
2. $\mathbf{k}_t^R$（解耦的 RoPE Key，维度 $d_h n_h$ 实际上所有头共享？注意公式 (4) 中每个头的 $k_t^R$ 是相同的，因此只需缓存一份）

因此总缓存大小远小于标准 MHA（后者需要缓存 $2 \times n_h \times d_h$ 的完整 K 和 V）。

---

#### 六、设计要点总结

- **低秩联合压缩**：K 和 V 共享同一个压缩潜向量 $\mathbf{c}_t^{KV}$，通过不同的上投影矩阵 $W^{UK}$ 和 $W^{UV}$ 分别重构。
- **RoPE 解耦**：位置编码不压缩，而是单独生成 $\mathbf{k}_t^R$ 并与内容部分拼接。这样既保留了 RoPE 的相对位置能力，又不破坏低秩压缩的效率。
- **Query 压缩**：主要用于训练时减少激活内存，推理时 Query 不缓存，因此压缩收益主要体现在训练阶段。

- K向量的RoPE由输入 $h$ 而不是 $c$ 计算得到主要是因为MLA的低秩压缩与RoPE不兼容：加入了RoPE的矩阵无法融合，因为中间两个矩阵与token位置相关。具体来说，RoPE需要对每个不同位置$t$的token施加不同的旋转矩阵，如果直接将RoPE施加在压缩后的$c_t^{KV}$上然后通过上投影矩阵$W^{UK}$重构：
    - 旋转矩阵 $R_t$ 与上投影矩阵$W^{UK}$无法融合，因为$R_t$是位置相关的，$W^{UK}$是共享的，注意力的计算变得和位置差相关了，而我们需要的是固定的投影矩阵，所以MLA选择了增加维度，即从输入 $h$ 进行RoPE位置编码然后再与K进行拼接
    - 位置信息会在低维压缩空间中被“稀释”（压缩维度过低，难以同时保留语义和位置信息 ）
    - 无法复用缓存（用了一个token的$c_t^{KV}$之后无法直接用于另一个token）
    - 结论：RoPE与低秩压缩在数学形式上是冲突的，这与MHA完全不同

- 为什么K的RoPE必须共享，因为K向量在推理时必须具备一致性。历史token的K向量在生成时就已经确定，后面生成新token时会反复被调用。如果我们为每个头单独存储带RoPE的K向量，存储量将乘以$n_h$倍，直接破坏了MLA"压缩KV缓存"的核心目的。
- 为什么Q的RoPE不需要共享，Q向量在推理过程中不需要缓存——每生成一个新token，都会重新计算当前token的Q值，它只对当前token有意义，不会被未来token复用。

#### 对Partial RoPE的进一步理解
- Partial RoPE可以有两种含义：一是对Attention的Q、K加RoPE时，可以只对小部分维度加；二是可以考虑层间RoPE与NoPE交替出现，并且NoPE的层可以占多数
- 这种部分维度的策略也隐含在MLA的拼接设计中：每个注意力头的K向量由$[k_{t,i}^C;k_t^R]$组成，其中$K_{t,i}^C$占512维（内容），$k_t^R$仅占64维（位置）。对于多头来说，共享一个$k_t^R$也构成了Partial RoPE的一种实现


![alt text](figure/Tranformer架构扩展/MLA示意图.png)


### 扩展：GQA也可以看作低秩投影  
低秩投影指的是：用一个**低维表示**来近似或替代原本的高维表示，并通过一个线性投影矩阵在低维与高维之间转换。  

GQA 可以解释为：原本 MHA 中每个查询头都有一个独立的 $K$ 向量（共 $H$ 个），现在我们只保留 $G$ 个“基” $K$ 向量，然后每个查询头的 $K$ 由这些基通过**固定的线性组合**得到。这正是低秩分解的形式。 

#### 低秩投影的解释

我们可以定义一个投影矩阵 $\mathbf{P} \in \mathbb{R}^{H \times G}$，其元素为：

$$
P_{i,g} =
\begin{cases}
1 & \text{如果查询头 } i \text{ 属于组 } g \\
0 & \text{否则}
\end{cases}
$$

那么，GQA 中每个头的 Key 可以通过投影得到：

$$
\mathbf{K}_{\text{GQA\_per\_head}} = \mathbf{P} \cdot \mathbf{K}_{\text{GQA}} \in \mathbb{R}^{H \times d_k}
$$

也就是说，GQA 的每个头的 Key 是共享基向量的线性组合（这里组合系数是 0 或 1，即选择对应组的基）。

如果我们允许更一般的组合（如加权平均），则 $\mathbf{P}$ 可以是一个稠密的投影矩阵，而 GQA 是它的特例（稀疏的 one-hot 选择）。

#### 低秩的体现

- 完整的 MHA 允许每个头有任意的 Key 向量，因此 $\mathbf{K}_{\text{MHA}}$ 的**行秩**理论上可以达到 $H$。
- 而 GQA 中，所有头的 Key 向量都位于由 $G$ 个基向量张成的子空间中，因此行秩**不超过** $G$。
- 当 $G < H$ 时，这构成了一个低秩约束（秩为 $G$，远小于 $H$）。

总结：已经一句话概括，GQA 将 H 个头的 Key 矩阵通过一个常值投影矩阵（分组选择）压缩到 G 个基向量上，因此 Key 矩阵的秩从上限 H 降至 G，正是一种低秩投影。


---

## 模块二：前馈网络演进（FFN → GLU/SwiGLU → MoE）
前馈网络（FFN）是Transformer中负责特征变换的核心，占模型70%左右的参数量，演进的核心目标是**提升特征表达能力，同时控制计算量**。

### 1. 原始FFN
原始Transformer的FFN结构：`线性层 → ReLU激活 → 线性层`，公式：
$$FFN(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$
通常是先把特征维度扩大4倍，再压缩回原维度，用非线性激活提升表达能力。

**核心痛点**：ReLU激活的表达能力有限，特征过滤是固定的，无法动态调整信息的保留和过滤。

### 2. GLU（门控线性单元）→ SwiGLU（主流方案）
**解决的核心问题**：提升FFN的特征表达能力，让模型能动态过滤信息。
#### 原理：
GLU把输入分成两路，一路做线性变换，另一路经过门控激活函数，两路相乘得到输出，公式：
$$GLU(x) = (W_1 x) \odot \sigma(W_2 x)$$
- $\sigma$是sigmoid激活函数，作为门控，动态决定哪些信息可以通过；
- $\odot$是逐元素相乘。

#### 主流变体SwiGLU：
把sigmoid门控换成SiLU（Swish）激活，是现在的工业界标准：
$$SwiGLU(x) = (W_1 x) \odot \text{SiLU}(W_2 x)$$

**核心优势**：
- 门控机制让模型能动态调整信息的流通，特征表达能力远超原始FFN；
- 训练更稳定，效果更好，Llama、Qwen、GPT-NeoX等所有主流模型全部采用SwiGLU作为FFN的默认结构。

### 3. MoE（混合专家模型，Mixture of Experts）
**解决的核心问题**：打破「参数量越大，计算量越大」的魔咒，用更少的计算量实现更大的参数量和更好的效果。
#### 原理：
- 把原来的单个FFN层，拆分成N个独立的「专家网络」（每个专家都是一个独立的FFN），外加一个「路由器（Router）」；
- 每个token输入后，路由器计算每个专家的匹配得分，只选得分最高的K个专家（通常K=1或2）参与计算，其他专家完全不激活。

比如Mixtral 8x7B，有8个7B的专家网络，每个token只激活2个专家，实际计算量只有14B参数量的模型，但效果接近56B的密集模型。

**核心优势**：
- 参数量可以做得很大，但实际计算量很小，训练和推理成本大幅降低；
- 是现在开源大模型扩容的主流方案，Mixtral、DeepSeek-MoE、Qwen-MoE都采用这个架构。

---

## 模块三：归一化演进（LayerNorm → RMSNorm）
归一化是Transformer训练稳定的核心，演进的核心目标是**简化计算、提升训练速度，同时不损失效果**。

### 1. 原始LayerNorm（LN，层归一化）
原始Transformer的归一化方案，核心逻辑：对每个样本的所有特征做归一化，减去均值，除以标准差，公式：
$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
- $\mu$是特征的均值，$\sigma$是标准差；
- $\gamma$和$\beta$是可学习的缩放和平移参数。

**核心痛点**：
- 每次都要计算均值$\mu$，做中心化操作，增加了计算量；
- 均值计算在长序列、大batch场景下，会引入额外的计算开销。

### 2. RMSNorm（均方根归一化）
**解决的核心问题**：简化LayerNorm的计算，提升速度，同时保持训练稳定性。
#### 原理：
去掉了LayerNorm中均值中心化的操作，只保留均方根（RMS）做归一化，公式：
$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}}$$

**核心优势**：
- 计算量比LayerNorm减少40%左右，训练和推理速度更快；
- 去掉了均值中心化，反而提升了训练稳定性，效果和LayerNorm持平甚至更好，平方抑制了一场，更鲁棒，深层网络梯度更干净不易爆炸或消失；
- 现在所有主流大模型（Llama、Qwen、Mistral等）全部采用RMSNorm作为默认归一化方案。

---

## 模块四：位置编码演进（绝对位置编码 → RoPE → YaRN）
Transformer本身是无序的，位置编码负责给模型注入文本的顺序信息，演进的核心目标是**提升长文本泛化能力、扩展上下文窗口、不损失模型效果**。

### 1. 原始绝对正弦位置编码
原始Transformer的方案，用固定的正弦/余弦函数生成位置编码，直接和token的嵌入特征相加。
- 原始位置编码可以简单（为了简化理解假设Q和K都是单位矩阵，并不完全准确，比如Score计算还有softmax等没有这么简单）写成如下：
$$\begin{aligned}
Q &= W_q(X_m + P_m) \\
K &= W_k(X_n + P_n) \\
Score &= (X_m + P_m)(X_n + P_n)^T \\
&= (X_m + P_m)(X_n^T + P_n^T) \\
&= X_m X_n^T + X_m P_n^T + P_m X_n^T + P_m P_n ^ T
\end{aligned}$$
可以看到$X_m X_n^T$和$P_m P_n ^ T$分别是纯语义和纯相对位置计算，而$X_m P_n^T$和$P_m X_n^T$是交叉噪声

**核心痛点**：
- 泛化性极差，预训练时的上下文窗口是2k，推理时超过2k的位置，模型完全不认识，效果断崖式下跌；
- 注意力内积时之和Q、K内容相关，无法很好地捕捉相对位置关系，长文本效果差。

### 2. RoPE（旋转位置编码，Rotary Position Embedding）（[学习视频地址](https://www.bilibili.com/video/BV1FjrCBdESo/?spm_id_from=333.1391.0.0&vd_source=d3285a2ba86bc368a3901aac90d388ea)）
**解决的核心问题**：完美结合绝对位置编码和相对位置编码的优势，是现在大模型的事实标准。
#### 核心思想
RoPE 想做到：当对 Query 和 Key 向量应用一个与位置相关的变换后，它们的点积只依赖于位置的差值（相对位置），而不依赖绝对位置。

具体地，对于位置 $m$ 的向量 $\mathbf{x}_m$ 和位置 $n$ 的向量 $\mathbf{x}_n$，我们定义变换 $f$ 使得：

$$
\langle f(\mathbf{x}_m, m), f(\mathbf{x}_n, n) \rangle = g(\mathbf{x}_m, \mathbf{x}_n, m-n)
$$

即内积只与位置差有关。RoPE 通过将 $\mathbf{x}$ 的每一对维度看作一个复数，然后乘以一个与位置相关的旋转因子来实现。

#### Givens旋转矩阵的作用
在实数域中，Givens 旋转矩阵是用于对二维向量进行旋转的正交矩阵：

$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

它作用在二维向量 $[a, b]^T$ 上，结果是将该向量逆时针旋转 $\theta$ 角度。

RoPE 将原始的 $d$ 维向量（$d$ 为偶数）分成 $d/2$ 个二维子空间，每个子空间独立地应用一个旋转，旋转角度与该子空间的频率以及位置 $m$ 有关：

$$
\theta_k = m \cdot \omega_k,\quad \omega_k = 10000^{-2k/d},\quad k = 0,1,\dots,d/2-1
$$

于是，对第 $k$ 个子空间的二维向量 $[x_{2k}, x_{2k+1}]$，应用旋转矩阵 $R(m\omega_k)$。

#### 块对角矩阵是整体实现形式
将所有这些二维旋转矩阵沿主对角线排布，就得到一个块对角矩阵：

$$
\mathcal{R}_m =
\begin{bmatrix}
R(m\omega_0) & 0 & \dots & 0 \\
0 & R(m\omega_1) & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & R(m\omega_{d/2-1})
\end{bmatrix}
$$

这个矩阵的尺寸是 $d \times d$，除了主对角线上是若干个 $2 \times 2$ 的 Givens 旋转块，其余位置全是 0。

于是，位置编码后的向量为：

$$
f(\mathbf{x}_m, m) = \mathcal{R}_m \cdot \mathbf{x}_m
$$

由于 $\mathcal{R}_m$ 是正交矩阵且块对角，可以高效地分块计算，并且保持所有线性变换性质。

#### 逐元素的等价实现
由于 $\mathcal{R}_m$的稀疏性，直接用矩阵乘法会浪费算力，使用逐元素等价实现将计算复杂度从 $O(d^2)$ 降低到了 $O(d)$。  
观察 $\mathcal{R}_m$ 的结构：它只是将每连续两个维度视为一组，独立旋转。因此完全可以在二维子空间内单独计算，避免完整矩阵乘法。
逐元素（element-wise）的等价实现：

$$
\begin{bmatrix}
q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}
\end{bmatrix}
\otimes
\begin{bmatrix}
\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1}
\end{bmatrix}
+
\begin{bmatrix}
-q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d-1} \\ q_{d-2}
\end{bmatrix}
\otimes
\begin{bmatrix}
\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}
\end{bmatrix}
$$

- $\otimes$ 表示逐元素相乘（Hadamard product）。
- 第一项：原始向量与 $\cos$ 值逐元素乘。
- 第二项：对每个二维组 $(q_{2k}, q_{2k+1})$，构造 $(-q_{2k+1}, q_{2k})$（即旋转 90 度后的向量），再与 $\sin$ 值逐元素乘。

#### 从复数角度理解RoPE
对于位置 $m$ 的向量 $\mathbf{q}$，其第 $k$ 组 $(q_{2k}, q_{2k+1})$ 对应的复数为：
$$
q_k^{(c)} = q_{2k} + i q_{2k+1}
$$

RoPE 对该组施加的旋转角度为：
$$
\theta_k(m) = m \cdot \omega_k, \quad \text{其中 } \omega_k = 10000^{-2k/d}
$$

所以旋转后的复数：
$$
\tilde{q}_k^{(c)} = q_k^{(c)} \cdot e^{i m \omega_k}
$$

同样地，对 Key 向量第 $k$ 组的复数旋转：
$$
\tilde{k}_k^{(c)} = k_k^{(c)} \cdot e^{i n \omega_k}
$$

复数版本的点积定义为取实部：
$$
\langle \tilde{q}, \tilde{k} \rangle = \text{Re}\left( \sum_k \tilde{q}_k^{(c)} \cdot \overline{\tilde{k}_k^{(c)}} \right)
$$

计算：
$$
\tilde{q}_k^{(c)} \cdot \overline{\tilde{k}_k^{(c)}} = \left( q_k^{(c)} e^{i m \omega_k} \right) \cdot {\left( \overline{k_k^{(c)}} e^{-i n \omega_k} \right)} = q_k^{(c)} \overline{k_k^{(c)}} \cdot e^{i(m-n)\omega_k}
$$

取实部后，结果只依赖于 $m-n$，完美得到相对位置。


#### RoPE的简单直观理解
- RoPE编码简单理解（直接将位置信息注入到旋转矩阵的角度中，其中旋转矩阵的转置就是反方向的旋转）：
$$
\begin{aligned}
Q &= R_m X_m \\
K &= R_n X_n \\
\text{Score} &= (R_m X_m)(R_n X_n)^T \\
&= X_m (R_{n-m}) X_n^T
\end{aligned}
$$
具体计算：
$$
\begin{aligned}
q' &= R(m\theta)q \\
k' &= R(n\theta)k \\
\text{Score} &= (q')^T \cdot k' \\
&= (R(m\theta) \cdot q)^T \cdot (R(n\theta) \cdot k) \\
&= q^T \cdot R(m\theta)^T \cdot R(n\theta) \cdot k  \\
&= q^T \cdot R((n-m)\theta)^T \cdot k
\end{aligned}
$$
对于高维向量，每两个维度凑成一对二维向量，每对二维向量一起旋转，讲这些小旋转矩阵组合在一起（块对角矩阵）：
$$
\begin{pmatrix}
\cos\alpha & -\sin\alpha & 0 & 0 & \dots & 0 \\
\sin\alpha & \cos\alpha & 0 & 0 & \dots & 0 \\
0 & 0 & \cos\alpha & -\sin\alpha & \dots & 0 \\
0 & 0 & \sin\alpha & \cos\alpha & \dots & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & 0 & \dots & \begin{smallmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{smallmatrix}
\end{pmatrix}
$$
而对于不同组的旋转角度 $\theta$（和正余弦位置编码一样，低维旋转角度大，高维旋转角度小，每个子空间对应不同的旋转频率，学习不同尺度的相对位置模式）：
$$
\theta_i = 10000^{-\frac{2(i-1)}{d_{\text{model}}}} \\
i \in \left[1, 2, \dots, \frac{d}{2}\right]
$$



#### 总结原理：
- 不直接给特征加位置编码，而是通过旋转矩阵，对token的嵌入特征做旋转，用旋转的角度来表示token的绝对位置，角度差=相对位置，；
- 两个token的注意力得分，只和它们的相对位置差有关，天然支持相对位置建模。

**核心优势**：
- 长文本泛化能力远超绝对位置编码，预训练的上下文窗口内效果极好；
- 可以通过线性插值扩展上下文窗口，不用重新预训练就能支持更长的文本；
- 现在Llama、Qwen、Mistral、GPT-NeoX等几乎所有主流开源大模型，全部采用RoPE作为默认位置编码。

### RoPE外推
- 线性缩放（位置内插 Positional Interpolation）
- NTK-aware Scaled RoPE
- NTK-by-parts
- YaRN
- Leaky ReRoPE 和 ReRoPE

### 3. YaRN（Yet another RoPE extensioN）
**解决的核心问题**：RoPE线性插值扩展上下文时，会出现效果损失，需要大量微调，YaRN实现了「零微调/少量微调就能大幅扩展上下文窗口」。
#### 原理：
针对RoPE的旋转角度做了优化，对不同频率的位置编码做了差异化的缩放，解决了线性插值带来的低频信息损失问题。

**核心优势**：
- 不用重新预训练，仅需少量微调，就能把Llama 2 7B的4k上下文窗口扩展到128k，效果几乎没有损失；
- 是现在长上下文大模型的主流扩展方案，被Llama 3、Qwen2等模型广泛采用。

---

# 三、2025年主流大模型架构标配（一句话总结）
现在所有主流开源大模型（Llama 3、Qwen2、Mistral等）的架构，已经形成了统一的标准：
> **RMSNorm归一化 + RoPE旋转位置编码 + GQA分组查询注意力 + SwiGLU前馈网络**，长文本支持用YaRN扩展，低成本扩容用MoE架构。