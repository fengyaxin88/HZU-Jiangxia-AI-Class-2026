import torch

# 1. 创建Tensor
x = torch.tensor([1, 2, 3, 4])
x_float = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
x_zeros = torch.zeros((2, 3))
x_rand = torch.randn(2, 2)  # 标准正态分布

# 2. Tensor基本运算
y = torch.tensor([5, 6, 7, 8])
add_res = x + y
mul_res = x * y
mat_res = x_float @ x_float.T  # 矩阵乘法

# 3. Tensor索引与切片
slice_res = x[1:3]
select_res = x_float[x_float > 2.0]

# 4. Tensor维度变换
x_reshape = x_float.reshape(1, 4)
x_squeeze = x_reshape.squeeze()
x_unsqueeze = x_squeeze.unsqueeze(0)

# 5. 自动求导（autograd）
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a ** 2 + b * 3
c.backward()  # 反向传播求梯度

# 打印所有运行结果
if __name__ == "__main__":
    print("1. 创建Tensor：")
    print("整数Tensor：", x)
    print("浮点矩阵Tensor：", x_float)
    print("零矩阵：", x_zeros)
    print("随机Tensor：", x_rand)

    print("\n2. 基本运算：")
    print("加法：", add_res)
    print("乘法：", mul_res)
    print("矩阵乘法：", mat_res)

    print("\n3. 索引切片：")
    print("切片[1:3]：", slice_res)
    print("筛选>2.0：", select_res)

    print("\n4. 维度变换：")
    print("reshape(1,4)：", x_reshape)
    print("squeeze降维：", x_squeeze)
    print("unsqueeze升维：", x_unsqueeze)

    print("\n5. 自动求导：")
    print("c = a²+3b 的结果：", c)
    print("a的梯度(da/dc)：", a.grad)
    print("b的梯度(db/dc)：", b.grad)