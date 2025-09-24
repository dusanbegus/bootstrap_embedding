import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn
import sys
from qiskit import QuantumCircuit



def loss(c):
    state=np.array(c)
    qc=QuantumCircuit(2)
    qc.initialize(state, [0,1])
    print(qc.draw())

if __name__ == "__main__":
    loss([0,0,0,1])
    