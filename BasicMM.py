import numpy as np
import math

def cosd(a):
    return math.cos(math.radians(a))

def sind(a):
    return math.sin(math.radians((a)))

def Jog(theta):
    JogMatrix = np.array([[1, 0, 0, 0],
                          [0, cosd(2 * theta), -sind(2 * theta), 0],
                          [0, sind(2 * theta), cosd(2 * theta), 0],
                          [0, 0, 0, 1]])
    return JogMatrix

def Retarder(theta, delta):
    RetarderMM = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, cosd(delta), -sind(delta)],
                           [0, 0, sind(delta), cosd(delta)]])
    RetarderMM = np.dot(Jog(theta), np.dot(RetarderMM, Jog(-theta)))
    return RetarderMM

def Polarizer(theta):
    PolarizerMM = np.array([[1, 1, 0, 0],
                            [1, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
    PolarizerMM = np.dot(Jog(theta), np.dot(PolarizerMM, Jog(-theta)))
    return PolarizerMM