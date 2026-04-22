# 神经网络与PyTorch框架
## 张量
Numpy数组是Python中用于存储和处理多维数值数据的基础结构，所有元素必须是同一种数据类型，支持高效的矩阵运算、统计计算（高效的CPU数值计算，丰富的科学计算函数库，无法GPU加速，没有自动微分，无法直接用于神经网络训练）。
```
import numpy as np

# 1. 创建数组
a = np.array([1, 2, 3])  # 1维数组（向量）
b = np.array([[1, 2], [3, 4]])  # 2维数组（矩阵）
c = np.zeros((2, 3))  # 全0数组
d = np.random.rand(2, 3)  # 随机数组

# 2. 基础运算（元素级运算）
print("a + 1：", a + 1)  # [2, 3, 4]
print("b * 2：", b * 2)  # [[2,4],[6,8]]
print("矩阵乘法：", np.dot(b, b))  # [[7,10],[15,22]]

# 3. 统计计算
print("b的均值：", np.mean(b))  # 2.5
print("b的最大值：", np.max(b))  # 4
```
Tensor张量是PyTorch专门为深度学习设计的（同一种数据类型的）多维数值数据结构，不仅支持Numpy的所有数值运算，还能在**GPU上加速**，并且内置了**自动微分（Autograd）**功能（神经网络反向传播的核心），NumPy与PyTorch共享同一块内存（零拷贝转换）
```
import torch

# 1. 创建张量（和NumPy语法几乎一样）
a = torch.tensor([1, 2, 3])  # 1维张量
b = torch.tensor([[1, 2], [3, 4]])  # 2维张量
c = torch.zeros((2, 3))  # 全0张量
d = torch.rand((2, 3))  # 随机张量

# 2. 基础运算（和NumPy几乎一样）
print("a + 1：", a + 1)  # tensor([2, 3, 4])
print("b * 2：", b * 2)  # tensor([[2,4],[6,8]])
print("矩阵乘法：", torch.matmul(b, b))  # tensor([[7,10],[15,22]])

# 3. 核心功能1：放到GPU上（如果有GPU的话）
if torch.cuda.is_available():
    b_gpu = b.to("cuda")  # 张量移到GPU
    print("GPU上的张量：", b_gpu)
    print("GPU运算：", b_gpu * 2)  # 运算在GPU上执行，速度极快

# 4. 核心功能2：自动微分（Autograd）
# 创建张量时设置 requires_grad=True，自动追踪运算
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x²
y.backward()  # 反向传播，自动算梯度
print("x的梯度：", x.grad)  # tensor(4.)（dy/dx = 2x，x=2时梯度是4）
```
### 张量（默认的数据类型是float32）的基本创建方式
* torch.tensor 根据指定数据创建张量（更常用）
* torch.Tensor 根据形状创建张量，也可用来创建指定数据的张量
* torch.IntTensor、torch.FloatTensor、torch.DoubleTensor创建指定类型的张量
* 需要掌握的方式：tensor(值，类型)，eg torch.tensor(data, dtype=torch.float)
<br><br>

### 创建线性和随机张量
* torch.arange()（起始值、结束值、步长）和torch.linspace()（起始值、结束值、元素个数）创建线性张量
* torch.random.initial_seed() （基于当前时间戳生成）和 torch.random.manual_seed（）（根据传入的种子生成）随机种子设置
* torch.rand（均匀分布）/randn()(n表示正态分布)创建随机浮点类型张量
* torch.randint(low, high, size())创建随机整数类型张量

### 创建全0全1指定值张量
* torch.ones和torch.ones_like创建全1张量
    ```
    t1 = torch.ones(2, 3)
    t2 = torch.tensor([1,2], [3,4])
    t3 = torch.ones_like(t2)
    ```
* torch.zeros和torch.zeros_like创建全0张量
* torch.full和torch.full_like创建全为指定值张量  
    `t = torch.full(size=(2,3), fill_value=255)`

### 元素类型转换
* data.type(torch.DoubleTensor)（eg，t2 = t1.type(torch.int16）
* data.half/double/float/short/int/long()

### 张量的类型转换
张量转换为NumPy数组  
* 使用Tensor.numpy函数将张量转换为 ndarray 数组，但是共享内存（浅拷贝），可以使用copy函数避免共享（原因是tensor -> numpy()是浅拷贝，numpy() -> copy()是深拷贝，numpy库的copy()是深拷贝）
NumPy数组转换为张量
* 使用from_numpy函数将ndarray数组转换为Tensor，默认共享内存，使用copy函数避免共享
* 使用torch.tensor函数将ndarray数组转换为 Tensor，默认不共享内存
* 对于只有一个元素的标量张量，使用item()函数将该值从张量中提取出来

## 张量基本运算
加减乘除取负号：add、sub、mul、div、neg  
带下划线的版本会修改原数据（类似于inplace=True）:add_、sub_、mul_、div_、neg_
<br><br>


点乘运算  
* 点乘（Hadamard）指的是相同形状的张量对应位置的元素相乘，使用mul和运算符*实现

矩阵乘法运算  
矩阵乘法要求第一个矩阵 shape:(m, n)，第二个矩阵 shape(m, p)，两个矩阵点积运算 shape 为：(n, p)  
* 运算符@用于进行两个矩阵的乘积运算，torch.matmul对进行乘积运算的两矩阵形状没有限定。对于输入的shape不同的张量，对应的最后几个维度必须符合矩阵运算规则（对于高维张量，前面的Batch维度可以一致或者广播，最后两个维度必须满足基础矩阵乘法规则，即第一个张量的倒数第一维度=第二个张量的倒数第二维）（广播机制：维度不同时，在小维度前面补1，某个维度的大小是1时自动扩展成和另一个张量相同的大小）

常用运算函数  
* sum()、max()、min()、mean()  都有dim参数，0表示列，1表示行
* pow()、sqrt()、exp()、log()、log2()、log10() 都没有dim参数

### 张量的索引操作
对于`t = torch.randint(1, 10, (5, 5))`
* 简单行列索引，格式：张量对象[行， 列]  （eg，获取第2行数据 t[1] 或 t[1,:]，这里:表示所有列）
* 列表索引，前边的表示行，后边的表示列   （eg，t1[[1, 3], [2, 4]]表示取(1, 2)和(3, 4)两个位置的元素;t1[[[0], [1]], [1, 2]]表示获取第0, 1行的1, 2列的4个元素）
* 范围索引 （eg, t1[:3, :2]表示前3行前2列，t1[1:, :2]表示第2行到最后一行前2列的数据，t1[1::2, ::2]表示所有奇数行偶数列的数据）
* 布尔索引，t1[torch.tensor([True, False, False, True], :)]（eg, data[data[:, 2] > 5]表示第三列大于5的行数据，data[:, data[1]>5]表示第2行大于5的列数据）
* 多维索引,（eg, 获取0轴上的第一个数据data[0, :, :], 获取1轴上的第一个数据data[:, 0, :]，获取2轴上的第一个数据data[:, :, 0]）

### 张量的形状操作
t1.shape[0]获取第一个维度，t1.shape[-1]获取最后一个维度
* reshape函数，张量数据不变的情况下改变数据的维度，将其转换成指定的形状，不改变数据的顺序
<br><br>
* unsqueeze函数，升维，在指定维度插入一个大小为1的新维度。
    ```
    a=torch.randn(1,3)
    print(a.shape)
    b=torch.unsqueeze(a,0)  或 b = a.unsqueeze(0)
    print(b.shape)
    c=torch.unsqueeze(a,1)  或 c = a.unsqueeze(1)
    print(c.shape)
    d=torch.unsqueeze(a,2)  或 d = a.unsqueeze(2)
    print(d.shape)
    ```
* squeeze函数，降维，移除所有大小为1的维度，或者移除指定维度大小为1的维度
    ```
    x = torch.zeros(1, 3, 1, 5)    # shape: [1, 3, 1, 5]

    y = x.squeeze()               # 删除所有为1的维度 → shape: [3, 5]
    z = x.squeeze(0)              # 删除第0维（1） → shape: [3, 1, 5]
    w = x.squeeze(1)              # 尝试删除第1维（3），不成功 → shape: [1, 3, 1, 5]

    print(y.shape)  # torch.Size([3, 5])
    print(z.shape)  # torch.Size([3, 1, 5])
    print(w.shape)  # torch.Size([1, 3, 1, 5])
    ```
<br><br>

* transpose函数，交换张量形状的指定维度，一次只能交换2个维度，参数是需要交换的维度，不改变原始张量，返回新的张量
    ```
    t1 = torch.randint(1, 10, size=(2, 3, 4))
    t2 = t1.tranpose(0 ,2)

    t1的shape还是(2, 3, 4)
    t2的shape是(4, 3, 2)
    ```
* permute函数，一次交换更多维度，位置参数是新维度顺序的索引
    ```
        t1 = torch.randint(1, 10, size=(2, 3, 4))
        t3 = t1.permute(2, 0, 1)
        t3的shape是（4, 2, 3）
    ```
<br><br>

* view函数，用于修改连续的张量（在内存中的存储顺序和在张量中的逻辑顺序需要一致），张量经过transpose或者permute处理以后就无法使用view进行操作
    ```
    t1 = torch.randint(1, 10, size=(2, 3))
    t2 = t1.view(3, 2)   # t2的shape变成了(3, 2)同时存储顺序不变

    ```

* contiguous函数，把不连续的张量->连续的张量，即基于张量中显示顺序修改内存中的存储顺序。is_contiguous()判断张量是否连续
    ```
    t3 = t1.tanspose(0, 1)  # 交换t1的两个维度,t3的shape变为(3, 2)
    t3.is_contiguous() # False，不连续，无法使用view函数
    t5 = t3.contiguous().view(2, 3) # 将t3处理为连续张量再使用view函数即可，t5的shape是(2, 3)
    ```

### 张量拼接操作
* cat函数，将多个张量根据指定的维度拼接起来，不改变维度数（除了拼接的那个维度外，其他维度保持一致）
    ```
    t1 = torch.randint(1, 10, (2, 3))
    t2 = torch.randint(1, 10, (5, 3))
    t3 = torch.cat([t1, t2], dim=0)  # t3的shape是(7, 3)
    ```

* stack函数，在一个新的维度上连接一系列张量，会增加一个维度，所有输入张量形状必须完全相同
    ```
    # 两个形状完全相同的 1D 张量：(3,)
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    print("a的形状：", a.shape)  # torch.Size([3])
    print("b的形状：", b.shape)  # torch.Size([3])

    # 1. 在 dim=0（新维度）上拼接
    stack_dim0 = torch.stack([a, b], dim=0)
    print("dim=0拼接后形状：", stack_dim0.shape)  # torch.Size([2, 3])
    print("dim=0拼接后内容：\n", stack_dim0)
    # 输出：
    # tensor([[1, 2, 3],
    #         [4, 5, 6]])

    # 2. 在 dim=1（新维度）上拼接
    stack_dim1 = torch.stack([a, b], dim=1)
    print("dim=1拼接后形状：", stack_dim1.shape)  # torch.Size([3, 2])
    print("dim=1拼接后内容：\n", stack_dim1)
    # 输出：
    # tensor([[1, 4],
    #         [2, 5],
    #         [3, 6]])


    # 高维例子
    # 创建4个视角的图像数据：每个形状都是 (8, 3, 224, 224)
    view1 = torch.randn(8, 3, 224, 224)  # 视角1
    view2 = torch.randn(8, 3, 224, 224)  # 视角2
    view3 = torch.randn(8, 3, 224, 224)  # 视角3
    view4 = torch.randn(8, 3, 224, 224)  # 视角4
    # 在 dim=1 上插入新的视角维度
    stacked_dim1 = torch.stack([view1, view2, view3, view4], dim=1)
    print("dim=1 stack后形状：", stacked_dim1.shape)  # torch.Size([8, 4, 3, 224, 224])
    ```
## 自动微分
梯度基本计算
* PyTorch不支持向量张量对向量张量的求导，只支持标量张量对向量张量的求导(x如果是张量，y必须是标量才可以进行求导)
* 计算梯度：y.backward()，y是一个标量
* 获取x点的梯度值：x.grad，会累加上一次的梯度值
* 梯度下降法求最优解：w = w - r * grad（r是学习率，grad是梯度值），清空上一次的梯度值（如果不清空，每次反向传播以后会自动累加梯度值）
    ```
    梯度下降法伪代码：
    正向计算（前向传播）计算loss
    梯度清零 x.grad.zero_()
    反向传播 loss.sum().backward()
    梯度更新 w = w - r * grad
    ```
* 梯度计算注意点，不能将自动微分的张量转换成numpy数组，会发生报错，可以通过detach()方法实现（一个张量一旦设置了自动微分即requires_grad=True，这个张量就不能直接转成numpy的ndarray对象了，需要通过detach()函数解决），通过detach()函数拷贝一份张量，然后再转换
    ```
    t1 = torch.tensor([10, 20], requires_grad=True, dtype=torch.float)
    n2 = t1.detach().numpy()    # 使用detach()复制一份就可以转换成numpy
    # 需要注意的是使用detach()函数复制的变量共享内存即浅拷贝

    ```
* PyTorch模拟线性回归
    * 数据流：numpy -> tensor -> DataSet -> DataLoader
    * 数据训练流程伪代码
    ```
    # 构造数据集
    x, y, coef = create_dataset()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    # 构造模型
    model = nn.Linear(in_features=1, out_features=1)
    # 构造平方损失函数
    criterion = nn.MSELoss()
    # 构造优化函数
    optimizer = optim.SGD(params=model.parameters(),lr=1e-2)
    
    # 模型训练
    epochs = 100
    epoch_loss = []
    total_loss = 0.0
    train_sample = 0.0
    for _ in range(epochs):
        for train_x, train_y in dataloader:
            y_pred = model(train_x.type(torch.float32))
            loss = criterion(y_pred, train_y.reshape(-1, 1).type(float32))
            total_loss += loss.item()
            train_sample += 1
            optimizer.zero_grad()   # 梯度清零
            loss.backward()     # 自动微分（反向传播）
            optimizer.step  # 更新参数
        epch_loss.append(total_loss / train_sample)

    ```
## 神经网络基础
人工神经网络（Artificial Neural Netword, ANN）简称神经网络（NN）  
激活函数用于对每层的输出数据进行变换，进而为整个网络注入非线性因素，神经网络就可以拟合各种曲线了
* 没有引入非线性因素的网络等价于使用一个线性模型来拟合
* 通过给网络输出增加激活函数，实现引入非线性因素，使得网络模型可以逼近任意函数，提升网络对复杂问题的拟合能力
常见的激活函数
隐藏层优先使用顺序ReLU、LeakyReLU、PReLU、Tanh、Sigmoid  
输出层优先使用二分类Sigmiod、多分类Softmax
1. sigmoid（常用于二分类，将结果映射到0-1之间，适用于浅层神经网络）
$$f(x) = \frac{1}{1+e^{-x}}$$
$$f^\prime=f(x)(1-f(x))$$
2. tanh（将结果映射到-1-1之间，适用于浅层神经网络）
$$f(x)=\frac{1-e^{-2x}}{1+e^{-2x}}$$
$$f^\prime=1-f^2(x)$$
3. ReLU（默认情况下只考虑正样本，可缓解过拟合，计算量小，适合深层神经网络，可以用LeakyReLU、PReLU考虑正负样本）
$$f(x)=max(0, x)$$
$$f^\prime=0或1$$
4. softmax（用于多分类，映射成为0-1的值，累和为1）
$$sofrmax(z_i)=\frac{e^{z_i}}{\sum_{j}{e^{z_i}}}$$

### 参数初始化
需要初始化的参数主要是权重和偏执，偏执一般初始化为0  
参数初始化的作用：
* 防止梯度消失或爆炸
* 提高收敛速度
* 保持对称性破除
常见参数初始化方法：
* 随机初始化，在$(-\frac{1}{\sqrt{d}}, \frac{1}{\sqrt{d}})$均匀分布，d是神经元的输入数量
* 均匀分布初始化 nn.init.uniform_()
* 正态分布初始化 nn.init.normal_()
* 全0初始化 nn.init.zeros_()
* 全1初始化 ones_()
* 固定值初始化 constant_()
* Kaiming 初始化(kaiming + ReLU，专为ReLU设计)，也叫 HE 初始化。分为正态分布的HE初始化、均匀分布的HE初始化；
    * 正态分布的HE初始化(kaiming_normal_())，从[0, std]中抽取样本，std = sqrt(2 / fan_in)
    * 均匀分布的HE初始化(kaiming_uniform_())，从[-limit, limit]中的均匀分布中抽取样本，limit是sqrt(6/fan_in)
    * fan_in是输入层神经元的个数
* Xavier初始化（Glorot初始化）
    * 正态分布的Xavier初始化(xavier_normal_())，从[0, std]中抽取样本，std = sqrt(2/(fan_in + fan_out))
    * 均匀分布的Xavier初始化(xavier_uniform_())，[-limit, limit]中均匀分布抽取样本，limit是sqrt(6/(fan_in + fan_out))
    * fan_in 是输入层的神经元个数，fan_out是输出层的神经元个数

### 神经网络搭建
在pytorch中定义深度神经网络其实就是层堆叠的过程，继承自nn.Module，实现两个方法：
* __init__方法中定义网络中的层结构，主要是全连接层，并进行初始化
* forward方法，在实例化模型的时候，底层会自动调用该函数。该函数中初始化定义的layer传入数据，进行前向传播等。

### 损失函数
损失函数是用来衡量模型参数的质量的函数，衡量的方式是比较网络的输出和真实输出的差异。
* 多分类任务损失函数（使用nn.CrossEntropyLoss()实现），通常使用softmax将logits转换为概率的形式，多分类的交叉熵损失也叫softmax损失，计算方法（S是softmax激活函数）：
$$\mathcal{L} = -\sum_{i=1}^n y_i \log\left(S\left(f_\theta(\boldsymbol{x}_i)\right)\right)$$

* 二分类交叉熵损失函数（使用nn.BCELoss()实现），不再使用softmax激活函数，而是使用sigmoid激活数

$$L = -y \log \hat{y} - (1-y) \log (1-\hat{y})$$
* 回归任务MAE损失函数，也被称为 L1 Loss（nn.L1Loss），是以绝对值误差作为距离（由于L1 Loss具有稀疏性，为了惩罚较大的值，因此常将其作为正则项添加到其他Loss中作为约束）：
$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \left| y_i - f_\theta(x_i) \right|$$

* 回归任务MSE损失函数，也被称为L2 Loss（nn.MSELoss），以误差的平方和的均值作为距离损失函数（L2 Loss常作为正则项，当预测值与目标值相差很大时梯度容易爆炸）：
$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \left( y_i - f_\theta(x_i) \right)^2$$

* 回归任务Smooth L1损失函数（nn.SmoothL1Loss），光滑之后的L1。分段函数，[-1, 1]之间是L2损失，区间外是L1损失:
$$\text{smooth}_{L_1}(x)=
\begin{cases}
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$

### 网络优化方法
梯度下降算法
