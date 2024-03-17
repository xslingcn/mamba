import triton
import triton.language as tl
import torch
import math
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(rc={"figure.figsize": (10, 4)})
sns.set_style("whitegrid", {"axes.grid": False})
ones = (
    lambda *size: torch.ones(
        *size,
    )
    .float()
    .cuda()
)

zeros = (
    lambda *size: torch.zeros(
        *size,
    )
    .float()
    .cuda()
)

arrange = lambda n: torch.arange(n).float().cuda()

rand = lambda *size: torch.rand(*size).float().cuda()


def check(*inputs, prec=1e-4):
    for i, (a, b) in enumerate(zip(inputs[::2], inputs[1::2])):
        if isinstance(b, list):
            b = torch.tensor(b)
        c = torch.allclose(a.cpu(), b.cpu(), prec)
        c1 = torch.isclose(a.cpu(), b.cpu(), prec)
        assert c, f"check {i} failed, {a} \n {b} \n {c1}"
    print("Pass")


@triton.jit
def triton_hello_world(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    Ks = tl.arange(0, K)
    Ls = tl.arange(0, L)[:, None]
    x = tl.load(X + Ks)
    y = tl.load(Y + Ls)
    z = x + y
    tl.store(Z + Ls * K + Ks, z)


if __name__ == "__main__":
    x, y = arrange(4), ones(8, 4)
    z = zeros(8, 4)
    triton_hello_world[(1,)](x, y, z, 4, 8)
    print(z)
