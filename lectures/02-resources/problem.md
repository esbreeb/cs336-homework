# 第二讲作业：PyTorch 与资源核算

对应 [CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) Lecture 2（PyTorch, resource accounting）。讲义代码是 [`lecture_02.py`](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_02.py)。

这一讲还不用写 Transformer。要练的是两笔账：显存用多少字节，训练用多少 FLOPs。模型只到一个深层线性网络。

依赖 PyTorch。没有 GPU 也可以，自测都在 CPU 上跑。

```bash
python3 answer.py       # 答案模版，打印 ok
python3 resources.py    # 留空文件填完后也应打印 ok
```

## 三个文件

| 文件 | 做什么 |
| --- | --- |
| `problem.md` | 题目。 |
| [`answer.py`](answer.py) | 答案模版。完整可运行的实现。 |
| [`resources.py`](resources.py) | 留空的代码。把标了 `NotImplementedError` 的函数填上。 |

两个 Python 文件的接口一致。在编辑器里对它们做 diff，差的就是要写的函数体。不要改 `resources.py` 的函数签名和文末 `_self_check`。

## 餐巾纸上的两道题

FLOPs 是运算次数。FLOP/s 是硬件每秒能做多少次。两者读音一样，写的时候用斜杠把速率分开。

一次训练步里，矩阵乘法占掉绝大部分 FLOPs。`(m, n) @ (n, p)` 对每个 `(i, j, k)` 做一次乘法和一次加法，一共 `2·m·n·p`。元素加法、逐元素函数都是线性于张量大小，大矩阵上可以先忽略。

把形状换成「数据点数 × 参数量」：前向大约 `2 × tokens × parameters`。反向要为每一层多做两次同样规模的乘法（参数梯度和往下传的激活梯度），所以是前向的两倍，`4 × tokens × parameters`。加起来是

```text
一次训练步的 FLOPs = 6 × tokens × parameters
```

H100 的 bf16 宣传峰值是 `1979e12` FLOP/s，这个数字含 2:4 结构化稀疏。稠密训练用它的一半。MFU 是实际达到的 FLOP/s 除以宣传峰值。讲座里的例子取 MFU = 0.5。

因此：70B 参数、15T token、1024 张 H100，天数等于总 FLOPs 除以（稠密峰值 × 0.5 × 1024 × 86400）。结果四舍五入是 144 天。

显存按每个参数占多少字节来估。fp32 的 AdamW 是参数 4、梯度 4、两份矩估计各 4，共 16 字节。8 张 80GB 的 H100 一共 `8 × 80e9 / 16 = 40e9` 个参数。这个数不含激活。换成 bf16 参数和梯度，再留一份 fp32 主副本，加上两份 fp32 矩估计，仍然是 16 字节：更快，但不省显存。

## 精度和视图

fp32 每个数 4 字节，fp16 和 bf16 是 2 字节。fp16 的指数只有 5 位，`1e-8` 会下溢成 0。bf16 保留 fp32 的 8 位指数，同一个数不会变成 0，代价是尾数更粗。

张量是一块存储加上步长。`transpose` 和切片是新的视图，底层指针不变，改一边另一边也会变。不连续的视图不能再 `view`；`contiguous()` 会复制一份，于是不再共享存储。

## 模型和优化器

`Linear` 的权重用 `torch.randn(input_dim, output_dim) / sqrt(input_dim)`，这样输出的尺度不跟着输入维度一起涨。`Cruncher` 有 `num_layers` 个 `Linear(dim, dim)`，再加一个 `Linear(dim, 1)`，前向最后 `squeeze(-1)`，输出形状是 `(batch,)`。`num_layers = 0` 时只剩这个头，参数个数是 `dim`。

方阵层的参数是 `dim² × num_layers`，头还有 `dim` 个。激活按讲座的算法：每个样本、每个方阵层留一个长度为 `dim` 的向量，标量头不计入，所以是 `batch × dim × num_layers`。AdaGrad 为每个参数存一份 `g2`，梯度也是每个参数一份。fp32 下一次训练的字节数是

```text
4 × (参数 + 激活 + 梯度 + 优化器状态)
```

`D = 4`、`L = 2`、`B = 2` 时，参数 36、激活 16、梯度 36、优化器状态 36，一共 496 字节。同一次步的 FLOPs 是 `6 × 2 × 36 = 432`。

SGD 做 `p <- p - lr * grad`。AdaGrad 先把梯度平方累进 `g2`，再做 `p <- p - lr * grad / sqrt(g2 + 1e-5)`。`g2` 放在优化器自己的 `state` 里，跨 step 保留。

## 要写的代码

在 [`resources.py`](resources.py) 里实现：

- `tensor_nbytes`、`matmul_flops`、`model_flops`
- `h100_dense_bf16_flop_per_sec`、`training_days`
- `bytes_per_parameter`、`parameter_capacity`
- `shares_storage`
- `Linear`、`Cruncher`、`count_parameters`
- `cruncher_activation_count`、`training_memory_bytes`
- `SGD.step`、`AdaGrad.step`

`bytes_per_parameter` 只接受 `"fp32_adamw"` 和 `"mixed_bf16_adamw"`。
