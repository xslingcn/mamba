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
    # preparing initial values
    h[1, 1:] = h[0].cumsum(0)[:-1]  # ?? using torch cumsum
    # slicing understanding: h[1][1...len] = h[0].cumsum(0)[0...len-1]
    cumsum_tt[(_BLOCKS,)](x, h[1], y, h[1], K)


cumsum_block(x, y, K)
h_, y_ = cumsum(x.tolist())
check(y, y_)

# EMA: h(k)=α h(k−1)+(1−α)x(k)
#      y(k) = h(k)
alpha = 0.9


def ema(x, alpha):
    y = []
    h = 0
    for xi in x:
        h = alpha * h + (1 - alpha) * xi
        y.append(h)
    return h, y


x = arange(SEQLEN)
h_, y_ = ema(x.tolist(), alpha)


# Discrete-time state-space model scan (SSM):
# EMA allowing for variable coefficients
#       h(k) = a h(k−1) + b x(k)
#       y(k) = c h(k)
# A "dynamic-forgetting EMA"
def ssm_scan(x, a, b, c):
    y = []
    h = 0
    for xi in x:
        h = a * h + b * xi
        y.append(c * h)
    return h, y


h_, y_ = ssm_scan(x.tolist(), alpha, 1 - alpha, 1)


# Want to make SSM discrete
def op(a, b):
    return (a[0] * b[0], b[0] * a[1] + b[1])


"""
It 0:
    h = alpha * a, b * x[0] ~ h[0]
    y = c * h[0]
It 1:
    h = alpha * a * a, a * h[0] + b * x[1] ~ h[1]
    y = c * h[1]
It 2:
    h = alpha * a * a * a, a * h[1] + b * x[2] ~ h[2]
    y = c * h[2]
"""


def ssm_associative(x, a, b, c):
    y = []
    h = (alpha, 0)
    for k in range(len(x)):
        h_new = (a, b * x[k])
        h = op(h, h_new)
        y.append(c * h[1])
    return h, torch.stack(y)


assert ema(x, alpha)[0] == ssm_associative(x, alpha, 1 - alpha, 1)[0][1]
