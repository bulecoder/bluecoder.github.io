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
