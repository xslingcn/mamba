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

arange = lambda n: torch.arange(n).float().cuda()

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
    Ls = tl.arange(0, L)[:, None]  # add one dimension
    x = tl.load(X + Ks)
    y = tl.load(Y + Ls)
    z = x + y
    tl.store(Z + Ls * K + Ks, z)


x, y = arange(4), ones(8, 4)
z = zeros(8, 4)
print(f"x: {x} \ny: {y}\n z: {z}")
triton_hello_world[(1,)](x, y, z, 4, 8)
print(z)


@triton.jit
def triton_midres_check(X, Y, Z, K: tl.constexpr):
    Ks = tl.arange(0, K)
    x = tl.load(X + Ks)
    tl.store(Y + Ks, Ks)
    tl.store(Z + Ks, x)


x = rand(4)
print(f"midres_check: \nx: {x}")
ks = zeros(4)
xx = zeros(4)
triton_midres_check[(1,)](x, ks, xx, 4)
print(f"ks: {ks} \nxx: {xx}")


@triton.jit
def triton_hello_world_block(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    pid = tl.program_id(0)
    lid = pid * L
    Ks = tl.arange(0, K)
    Ls = tl.arange(0, L)[:, None]
    x = tl.load(X + Ks)
    y = tl.load(Y + (Ls + lid) * K + Ks)
    z = x + y
    tl.store(Z + (Ls + lid) * K + Ks, z)


x, y = arange(4), ones(8, 4)
z = zeros(8, 4)
print(f"x: {x} \ny: {y}\n z: {z}")
triton_hello_world_block[(1,)](x, y, z, 4, 8)
print(z)


def cumsum(x):
    y = []
    h = 0
    for xi in x:
        h += xi
        y.append(h)
    return h, y


@triton.jit
def plus_fn(a, b):
    return a + b


@triton.jit
def comsum_tt(X, Y, H, K: tl.constexpr):
    Ks = tl.arange(0, K)
    x = tl.load(X + Ks)
    hs = tl.associative_scan(x, 0, plus_fn)  # scan length Ks starting from X
    y = hs
    tl.store(Y + Ks, y)
    tl.store(H + Ks * 0, hs, mask=Ks == (K - 1))


K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS
x = arange(SEQLEN)
y = zeros(SEQLEN)
h = zeros(1)
comsum_tt[1,](x, y, h, K)
h_, y_ = cumsum(x[:K].tolist())
check(h[0], [h_], y[:K], y_)


@triton.jit
def cumsum_tt(X, H_0, Y, H, K: tl.constexpr):
    pid = tl.program_id(0)
    kid = K * pid
    Ks = tl.arange(0, K)
    x = tl.load(X + kid + Ks)
    h_0 = tl.load(H_0 + Ks * 0 + pid, Ks == 0, 0)
    x = h_0 + x  # vector addition
    hs = tl.associative_scan(x, 0, plus_fn)
    y = hs
    tl.store(Y + Ks + kid, y)
    tl.store(H + Ks * 0 + pid, hs, mask=Ks == (K - 1))


print(f"cumsum_tt\nx: {x}\n")
h = zeros(BLOCKS)
# h_0 is all zeros, so h is sums within each block
cumsum_tt[(BLOCKS,)](x, h, y, h, K=K)
print(f"h: {h}\n")
h_, y_ = cumsum(x[K * 2 : K * 3].tolist())
print(f"h_: {h_}\n")


def cumsum_block(x, y, K):
    seqlen = y.shape[0]
    _BLOCKS = seqlen // K
    h = zeros(2, _BLOCKS)
    cumsum_tt[(_BLOCKS,)](x, h[0], y, h[0], K)
    h[1, 1:] = h[0].cumsum(0)[:-1]  # ?? using torch cumsum
    # slicing understanding: h[1][1...len] = h[0].cumsum()[0...len-1]
    cumsum_tt[(_BLOCKS,)](x, h[1], y, h[1], K)


cumsum_block(x, y, K)
h_, y_ = cumsum(x.tolist())
check(y, y_)
