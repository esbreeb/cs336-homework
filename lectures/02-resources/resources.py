"""第二讲作业的留空代码。题目在 problem.md，完整实现在 answer.py。

只实现标了 NotImplementedError 的函数，不要改函数签名和文末自测。
运行：python3 resources.py
通过时打印 ok。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def tensor_nbytes(num_elements: int, dtype: torch.dtype) -> int:
    """元素个数乘每个元素的字节数。fp32 是 4，fp16 和 bf16 是 2。"""
    raise NotImplementedError


def matmul_flops(m: int, n: int, p: int) -> int:
    """(m, n) @ (n, p) 的 FLOPs。每个 (i, j, k) 一次乘法和一次加法。"""
    raise NotImplementedError


def model_flops(num_tokens: float, num_parameters: float) -> float:
    """一次训练步的模型 FLOPs：前向 2、反向 4，合计 6 × tokens × parameters。"""
    raise NotImplementedError


def h100_dense_bf16_flop_per_sec() -> float:
    """H100 bf16 稠密峰值。宣传的 1979e12 含 2:4 稀疏，稠密是它的一半。"""
    raise NotImplementedError


def training_days(
    num_parameters: float,
    num_tokens: float,
    num_gpus: int,
    peak_flop_per_sec: float,
    mfu: float,
) -> float:
    """总 FLOPs / (单卡峰值 × MFU × 卡数 × 每天秒数)。"""
    raise NotImplementedError


def bytes_per_parameter(plan: str) -> int:
    """每个参数在训练时占用的字节，不含激活。

    fp32_adamw：参数 4 + 梯度 4 + Adam 的两份状态各 4。
    mixed_bf16_adamw：bf16 参数 2 + bf16 梯度 2 + fp32 主副本 4 + 两份 fp32 状态各 4。
    """
    raise NotImplementedError


def parameter_capacity(total_bytes: float, bytes_per_parameter: float) -> float:
    """在给定显存和每参数字节数下，能放下的参数个数。不含激活。"""
    raise NotImplementedError


def shares_storage(x: torch.Tensor, y: torch.Tensor) -> bool:
    """两个张量是否指向同一块底层存储。view 和 transpose 共享，contiguous 复制之后不共享。"""
    raise NotImplementedError


class Linear(nn.Module):
    """y = x @ weight。weight 用 N(0, 1/input_dim)，使输出尺度不随输入维度变大。"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Cruncher(nn.Module):
    """L 层方阵线性层，再加一个映到标量的头。输出形状是 (batch,)。"""

    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    raise NotImplementedError


def cruncher_activation_count(batch_size: int, dim: int, num_layers: int) -> int:
    """每个样本、每个方阵层保留一个长度为 dim 的激活。最后的标量头不计入。"""
    raise NotImplementedError


def training_memory_bytes(
    num_parameters: int,
    num_activations: int,
    num_gradients: int,
    num_optimizer_states: int,
    bytes_per_value: int = 4,
) -> int:
    """参数、激活、梯度、优化器状态的字节数之和。"""
    raise NotImplementedError


class SGD(torch.optim.Optimizer):
    """p <- p - lr * grad。没有动量。"""

    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        raise NotImplementedError


class AdaGrad(torch.optim.Optimizer):
    """g2 <- g2 + grad^2，然后 p <- p - lr * grad / sqrt(g2 + 1e-5)。"""

    def __init__(self, params, lr: float = 0.01):
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        raise NotImplementedError


def _self_check() -> None:
    assert torch.tensor(1e-8, dtype=torch.float16).item() == 0.0
    assert torch.tensor(1e-8, dtype=torch.bfloat16).item() != 0.0

    assert tensor_nbytes(4 * 8, torch.float32) == 128
    assert tensor_nbytes(4 * 8, torch.float16) == 64
    assert tensor_nbytes(4 * 8, torch.bfloat16) == 64
    assert tensor_nbytes(12288 * 4 * 12288, torch.float32) == 2304 * 1024 * 1024

    assert matmul_flops(4, 8, 2) == 128
    assert model_flops(2, 36) == 432
    assert model_flops(15e12, 70e9) == 6.3e24

    assert h100_dense_bf16_flop_per_sec() == 1979e12 / 2
    days = training_days(
        num_parameters=70e9,
        num_tokens=15e12,
        num_gpus=1024,
        peak_flop_per_sec=h100_dense_bf16_flop_per_sec(),
        mfu=0.5,
    )
    assert math.isclose(days, 143.92659381842682, rel_tol=1e-9)
    assert round(days) == 144

    assert bytes_per_parameter("fp32_adamw") == 16
    assert bytes_per_parameter("mixed_bf16_adamw") == 16
    assert parameter_capacity(8 * 80e9, bytes_per_parameter("fp32_adamw")) == 40e9

    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = base.transpose(0, 1)
    assert shares_storage(base, transposed)
    assert not transposed.is_contiguous()
    copied = transposed.contiguous()
    assert not shares_storage(base, copied)
    base[0, 0] = 100
    assert transposed[0, 0].item() == 100

    torch.manual_seed(0)
    layer = Linear(3, 2)
    torch.manual_seed(0)
    expected_weight = torch.randn(3, 2) / math.sqrt(3)
    assert torch.allclose(layer.weight, expected_weight)

    model = Cruncher(dim=2, num_layers=1)
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.final.weight.copy_(torch.tensor([[2.0], [3.0]]))
    batch = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    assert torch.allclose(model(batch), torch.tensor([5.0, 6.0]))

    wide = Cruncher(dim=4, num_layers=2)
    sizes = [(name, param.numel()) for name, param in wide.state_dict().items()]
    assert sizes == [("layers.0.weight", 16), ("layers.1.weight", 16), ("final.weight", 4)]
    assert count_parameters(wide) == 36
    assert count_parameters(Cruncher(dim=4, num_layers=0)) == 4
    assert cruncher_activation_count(batch_size=2, dim=4, num_layers=2) == 16
    assert training_memory_bytes(36, 16, 36, 36) == 496

    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    sgd = SGD([weight], lr=0.1)
    weight.grad = torch.tensor([0.5, -1.0])
    sgd.step()
    assert torch.allclose(weight, torch.tensor([0.95, 2.1]))

    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    adagrad = AdaGrad([weight], lr=0.1)
    grads = [torch.tensor([0.5, -1.0]), torch.tensor([0.25, 0.5])]
    expected = torch.tensor([1.0, 2.0])
    second_moment = torch.zeros_like(expected)
    for grad in grads:
        weight.grad = grad.clone()
        adagrad.step()
        second_moment = second_moment + grad**2
        expected = expected - 0.1 * grad / torch.sqrt(second_moment + 1e-5)
    assert torch.allclose(weight.detach(), expected)

    torch.manual_seed(0)
    dim = 4
    true_w = torch.arange(dim, dtype=torch.float32)
    learned = Cruncher(dim=dim, num_layers=0)
    optimizer = SGD(learned.parameters(), lr=0.05)
    data = torch.randn(64, dim)
    target = data @ true_w

    def mse() -> float:
        return torch.nn.functional.mse_loss(learned(data), target).item()

    before = mse()
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(learned(data), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert mse() < before * 0.5

    print("ok")


if __name__ == "__main__":
    _self_check()
