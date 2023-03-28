import numpy as np
import math
import BasicMM

def InstrumentMM(N,
                 theta0, theta1, theta2, theta3,
                 delta1, delta2,
                 dtheta0, dtheta1, dtheta2, dtheta3,
                 ddelta1, ddelta2):
    Inst = np.zeros((N, 16))
    for i in range(N):
        tempA = np.dot(BasicMM.Polarizer(theta3[i] + dtheta3), BasicMM.Retarder(theta2[i] + dtheta2, delta2 + ddelta2))
        tempA = tempA[0, :]

        tempG = np.dot(BasicMM.Retarder(theta1[i] + dtheta1, delta1 + ddelta1), BasicMM.Polarizer(theta0[i] + dtheta0))
        tempG = tempG[:, 0]

        tempInst = np.kron(tempA, tempG.T)
        Inst[i, :] = tempInst
    return Inst.astype(np.float32)

def FourierBaseMatrix(step, N):
    stepinRad = math.radians(step)
    F = np.zeros((N, 25))
    for i in range(25):
        if i == 0:
            F[:, i] = 1
        elif i <= 12:
            F[:, i] = np.cos(np.arange(0, N) * stepinRad * 2 * i)
        else:
            F[:, i] = np.sin(np.arange(0, N) * stepinRad * 2 * (i - 12))
    PinvF = F.T / N * 2
    PinvF[0, :] = PinvF[0, :] / 2
    return [F.astype(np.float32), PinvF.astype(np.float32)]