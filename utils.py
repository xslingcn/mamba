import torch

K = 16
BLOCKS = 8
SEQLEN = K * BLOCKS

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
