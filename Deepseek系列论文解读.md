# DeepSeek系列论文解读

## KV Cache、MQA、GQA、MLA


[解读视频地址1](https://www.bilibili.com/video/BV1MZXPYNESh/?spm_id_from=333.788.videopod.episodes&vd_source=d3285a2ba86bc368a3901aac90d388ea&p=6)

[解读视频地址2](https://www.bilibili.com/video/BV1BYXRYWEMj/?spm_id_from=333.337.search-card.all.click&vd_source=d3285a2ba86bc368a3901aac90d388ea)


[解读视频地址3](https://www.bilibili.com/video/BV1n7rvYCEVT/?spm_id_from=333.1391.0.0&vd_source=d3285a2ba86bc368a3901aac90d388ea)



## DeepSeek v4
- MHC，在HC(超连接)的基础上面进行改进，因为HC在训练的时候会导致梯度爆炸，引入了双随机矩阵。工程实现上面（不能把必须是双随机矩阵直接写进反向传播，否则会破坏梯度），先让网络自由生成原始的矩阵，用Sinkhorn-Knoppo算法迭代将其变成离它最近的双随机矩阵去做残差更新