import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import scipy.signal as signal
import scipy.io as scio
import torch
import torch.optim
from torch.autograd import Variable

import Interp
import Gray2MM

Epochs = 200
N = 60
device = 'cuda'
method = 'Bicubic'
lr = 0.01

Img = scio.loadmat(r'IntensityImage.mat')['IntensityImage']

Filter = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
Filter = Filter * 1 / 9

Img_filtered = np.zeros((Img.shape[0] - 2, Img.shape[1] - 2, N))
for i in range(N):
    Img_filtered[:, :, i] = signal.convolve2d(Img[:, :, i], Filter, mode='valid')

I = Img_filtered.astype(np.float32)
I = np.swapaxes(I, 0, 2)
I = np.swapaxes(I, 1, 2)
I = I[np.newaxis, :]
I = torch.tensor(I, dtype=torch.float32, requires_grad=False, device=device)

Img = Img.astype(np.float32)
Img = np.swapaxes(Img, 0, 2)
Img = np.swapaxes(Img, 1, 2)
Img = Img[np.newaxis, :]
Img = torch.tensor(Img, dtype=torch.float32, requires_grad=False, device=device)

dx = torch.zeros(N, 1, dtype=torch.float32, requires_grad=True, device=device)
dy = torch.zeros(N, 1, dtype=torch.float32, requires_grad=True, device=device)
dx = Variable(dx, requires_grad=True)
dy = Variable(dy, requires_grad=True)

[F, PinvF] = Gray2MM.FourierBaseMatrix(step=3, N=N)
OrthF = np.eye(N) - np.dot(F, PinvF)
OrthF = OrthF.T
OrthF = torch.tensor(OrthF, dtype=torch.float32, requires_grad=False, device=device)
OrthF = torch.reshape(OrthF, (N, N, 1, 1))

Inst = Gray2MM.InstrumentMM(N, np.zeros((N, 1)), 3 * np.arange(0, N), 15 * np.arange(0, N), np.zeros((N, 1)),
               90, 90,
               0, 0, 0, 0,
               0, 0)
PinvInst = np.linalg.pinv(Inst)
OrthInst = np.eye(N) - np.dot(Inst, PinvInst)
OrthInst = OrthInst.T
OrthInst = torch.tensor(OrthInst, dtype=torch.float32, requires_grad=False, device=device)
OrthInst = torch.reshape(OrthInst, (N, N, 1, 1))

opt = torch.optim.Adam([dx, dy], lr=lr, betas=(0.8, 0.8))
LossHistory = []

for i in range(Epochs):
    [kernelx, kernely] = Interp.BindKernelGroup(dx, dy, method=method, device=device)

    Reconstructed_x = torch.nn.functional.conv2d(I, kernelx, stride=1, padding='valid', groups=N)
    Reconstructed = torch.nn.functional.conv2d(Reconstructed_x, kernely, stride=1, padding='valid', groups=N)
    Proj = torch.nn.functional.conv2d(Reconstructed, OrthF, stride=1, padding='valid', groups=1)
    L = torch.mean(torch.norm(Proj, 2, 1) / torch.norm(Reconstructed, 2, 1))

    opt.zero_grad()
    L.backward()
    opt.step()
    LossHistory.append(L.data.cpu())

[kernelx, kernely] = Interp.BindKernelGroup(dx, dy, method=method)
Reconstructed = torch.nn.functional.conv2d(Img, kernelx, stride=1, padding='valid', groups=N)
Reconstructed = torch.nn.functional.conv2d(Reconstructed, kernely, stride=1, padding='valid', groups=N)

Iout = Reconstructed.data.cpu().numpy()
scio.savemat('I.mat', {'I': Iout})

