import triton
import triton.language as tl
import torch
from utils import *

# EMA: h(k)=α h(k−1)+(1−α)x(k)
#      y(k) = h(k)
alpha = 0.9

x = arange(SEQLEN)
y = zeros(SEQLEN)
h = zeros(1)


def ema(x, alpha):
    y = []
    h = 0
    for xi in x:
        h = alpha * h + (1 - alpha) * xi
        y.append(h)
    return h, y


h_, y_ = ema(x.tolist(), alpha)


# State-space model scan (SSM):
# EMA allowing for different coefficients (but time independent!)
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


# Want to make SSM associative
# Now similar to a cumsum
def op(a, b):
    return (a[0] * b[0], b[0] * a[1] + b[1])


"""
f = h[0], h = h[1]
It 0:
    f = a * a
    h = a * 0 + b * x[0] ~ h[0]
    y = c * h[0]
It 1:
    f = a * a * a
    h = a * (b * x[0]) + b * x[1] ~ h[1]
    y = c * h[1]
It 2:
    f = a * a * a * a
    h = a * (a * (b * x[0]) + b * x[1]) + b * x[2] ~ h[2]
    y = c * h[2]
It 3:
    f = a * a * a * a * a
    h = a * (a * (a * (b * x[0]) + b * x[1]) + b * x[2]) + b * x[3] ~ h[3]
    y = c * h[3] 
      = c * a * a * a * b * x[0] + c * a * a * b * x[1] + c * a * b * x[2] + c * b * x[3]
...
It n:
    y = c * a * a * a * ... * a * b * x[0] + c * a * a * a * ... * b * x[1] + ... + c * b * x[n]
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


# Now make it parrallel
# same as op(a, b)
@triton.jit
def first_order_op(fl, xl, fr, xr):
    f = fl * fr
    x = fr * xl + xr
    return f, x


@triton.jit
def ssm_load(Ks, A, B, C):
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c


@triton.jit
def simple_ssm_tt(X, A, B, C, Y, K: tl.constexpr):
    Ks = tl.arange(0, K)

    pid = tl.program_id(0)
    lid = pid * K
    x = tl.load(X + lid + Ks)
    a, b, c = ssm_load(lid + Ks, A, B, C)

    h1, h2 = tl.associative_scan((a, b * x), 0, first_order_op)
    y = c * h2
    tl.store(Y + lid + Ks, y)


h = torch.zeros(2, BLOCKS).float().cuda()
a, b, c = ones(SEQLEN) * alpha, ones(SEQLEN) - alpha, ones(SEQLEN)
simple_ssm_tt[(1,)](x, a, b, c, y, K)
h_, y_ = ema(x[:K].tolist(), alpha)
check(y[:K], y_)
