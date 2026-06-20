## qkv_svd


### 项目简介：
对单头attention机制做svd奇异值分解，验证d_k维度作为秩上界对attention机制表达能力的硬约束

### 理论基础：
1，W_Q / W_K 的形状是 (embedding_dim, d_k)，因而有rank(W_Q), rank(W_K) ≤ d_k
2，对任意矩阵 A、B：rank(AB) ≤ min(rank(A), rank(B))，代入得rank(W_Q @ W_K.T) ≤ rank(W_Q) ≤ d_k

### 实验设计：
- 模型：单头 Attention，无位置编码，无 FFN
- 任务：复制任务（输入序列即标签）
- 控制变量：固定其他超参数，改变 d_k ∈ {16, 32, 64, 128}
- 分析对象：训练后的 W_Q @ W_K.T 的奇异值谱

### 实验结论：
d_k的维度虽然通过mn*nk=mk的方式消掉了，但是它始终作为秩隐含在结果当中，通过svd，它作为奇异值又重新显式暴露了出来。即使模型的矩阵乘法规模再大，无论训练多少轮，梯度如何更新，它作为秩上界是无法突破的，它的维度成为qkv模型表达能力的硬约束，也就是限制在d_k的子空间维度内。

### 实验环境：
- Python 3.x
- PyTorch
- matplotlib

