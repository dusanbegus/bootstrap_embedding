import optimizer
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn
import sys
from qiskit import QuantumCircuit
from qiskit.circuit.library import CSwapGate
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import functions


def gradient_descent(c, steps=1000):
    """
    Performs gradient descent optimization on the parameters.

    Parameters:
    c (torch.Tensor): Initial parameters to be optimized.
    steps (int): Number of optimization steps (iterations).

    Returns:
    torch.Tensor: Optimized parameters after the specified number of steps.
    """
    for step in range(steps):
        c_grad = optimizer.gradient(c)
        c = optimizer.optimizer_step(c, step, c_grad)
        if step % 1 == 0:
            print(f"Step {step}, Loss: {functions.loss(c)}")
    return c
if __name__ == '__main__':
    print(gradient_descent(torch.tensor(functions.create_random(), requires_grad=True), steps=50))
    sys.exit(0)