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

def optimizer_step(angles, eta, gradient_matrix):
    """
    Performs a single optimization step using the computed gradient.        
    Parameters:
    angles (torch.Tensor): Current parameters to be optimized.
    eta (float): Learning rate for the optimization step.
    gradient_matrix (torch.Tensor): Gradient of the loss with respect to parameters.
    Returns:


    torch.Tensor: Updated parameters after the optimization step.           
    """
    angles = angles - eta * gradient_matrix
    return angles

def gradient(c, angles, target, eta=0.001):
    """
    Computes the gradient of the loss with respect to parameters.

    Parameters:
    loss (torch.Tensor): The loss value.
    c (torch.Tensor): Parameters with respect to which the gradient is computed.

    Returns:
    torch.Tensor: Gradient of the loss with respect to parameters.
    """
    loss_function=functions.loss(c, angles, target_state=target)  
    grad=[]
    for alpha in angles:
        loss_alpha=functions.loss(c, angles+eta*torch.eye(len(angles))[list(angles).index(alpha)], target_state=target)
        delta_loss=loss_alpha-loss_function
        alpha_grad=delta_loss/eta
        grad.append(alpha_grad)
    alpha_g=torch.tensor(grad, requires_grad=True)
    return alpha_g

def main():
    return 0

if __name__ == '__main__':
    print(gradient(torch.tensor(functions.create_random()), torch.tensor(functions.create_init()),
                                        torch.tensor(functions.create_init())))
    sys.exit(main())



