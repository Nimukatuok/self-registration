import torch
import torch.nn as nn

def BindKernelGroup(dx, dy, method='Bilinear', device='cuda'):
    N = dx.shape[0]
    if(method == 'Bilinear'):
        p = 2
        InterpKernelFunc = LinearKernelFunc()
    elif(method == 'Bicubic'):
        a = -0.5
        p = 3
        InterpKernelFunc = CubicKernelFunc(a=a)
    elif(method == 'Lancozs'):
        a = 4
        p = 5
        InterpKernelFunc = LancozsKernelFunc(a=a)
    else:
        return

    kernelx = torch.zeros(N, 1, 1, 2 * p + 1, dtype=torch.float32, device=device)
    kernely = torch.zeros(N, 1, 2 * p + 1, 1, dtype=torch.float32, device=device)

    for i in range(N):

        tempx = torch.arange(-p, p + 1, device=device) + dx[i]
        tempy = torch.arange(-p, p + 1, device=device) + dy[i]

        tempx = torch.reshape(tempx, (-1, 1))
        Interpx = InterpKernelFunc(tempx)
        kernelx[i, 0, :, :] = Interpx.T

        tempy = torch.reshape(tempy, (-1, 1))
        Interpy = InterpKernelFunc(tempy)
        kernely[i, 0, :, :] = Interpy
    return [kernelx, kernely]

class LinearKernelFunc(nn.Module):
    def __init__(self):
        super(LinearKernelFunc, self).__init__()
        return

    def forward(self, x):
        i0 = x < -1
        i1 = (x >= -1) & (x < 0)
        i2 = (x >= 0) & (x < 1)
        i3 = x >= 1

        y0 = 0
        y1 = 1 + x
        y2 = 1 - x
        y3 = 0
        return i0 * y0 + i1 * y1 + i2 * y2 + i3 * y3

class CubicKernelFunc(nn.Module):
    def __init__(self, a=-0.5):
        super(CubicKernelFunc, self).__init__()
        self.a = a
        return

    def forward(self, x):
        i0 = x < -2
        i1 = (x >= -2) & (x < -1)
        i2 = (x >= -1) & (x < 0)
        i3 = (x >= 0) & (x < 1)
        i4 = (x >= 1) & (x < 2)
        i5 = x >= 2

        y0 = 0
        y1 = self.a * (-x) ** 3 - 5 * self.a * (-x) ** 2 + \
            8 * self.a * (-x) - 4 * self.a
        y2 = (2 + self.a) * (-x) ** 3 - (3 + self.a) * (-x) ** 2 + 1
        y3 = (2 + self.a) * x ** 3 - (3 + self.a) * x ** 2 + 1
        y4 = self.a * x ** 3 - 5 * self.a * x ** 2 + \
            8 * self.a * x - 4 * self.a
        y5 = 0
        return i0 * y0 + i1 * y1 + i2 * y2 + i3 * y3 + i4 * y4 + i5 * y5

class LancozsKernelFunc(nn.Module):
    def __init__(self, a=6):
        super(LancozsKernelFunc, self).__init__()
        self.a = a
        return

    def forward(self, x):
        i0 = x < -self.a
        i1 = (x >= -self.a) & (x < self.a)
        i2 = (x >= self.a)

        y0 = 0
        y1 = torch.sinc(x) * torch.sinc(x / self.a)
        y2 = 0
        return i0 * y0 + i1 * y1 + i2 * y2