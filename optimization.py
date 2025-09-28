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


def gradient_descent(c, angles, target, steps=1000, eta=0.01):
    """
    Performs gradient descent optimization on the parameters.

    Parameters:
    c (torch.Tensor): Initial parameters to be optimized.
    steps (int): Number of optimization steps (iterations).

    Returns:
    torch.Tensor: Optimized parameters after the specified number of steps.
    """
    for step in range(steps):
        angles_grad = optimizer.gradient(c,angles, target, eta)
        angles_f = optimizer.optimizer_step(angles, eta, angles_grad)
        if step % 1 == 0:
            print(f"Step {step}, Loss: {functions.loss(c, angles_f, target_state=target)}")
    return c,angles_f
if __name__ == '__main__':
    print(gradient_descent(torch.tensor(functions.create_random()), torch.tensor(functions.create_init()),
                                        torch.tensor(functions.create_init),
                                        steps=50))
    sys.exit(0)