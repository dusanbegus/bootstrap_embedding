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

def optimizer_step(c, step, gradient_matrix):
    """
    Performs a single optimization step using the Adam algorithm.

    Parameters:
    c (torch.Tensor): Current parameters to be optimized.
    step (int): Current optimization step (iteration).
    gradient_matrix (torch.Tensor): Gradient of the loss with respect to parameters.

    Returns:
    torch.Tensor: Updated parameters after the optimization step.
    """
    # Hyperparameters
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialize moment estimates
    if step == 0:
        m = torch.zeros_like(c)
        v = torch.zeros_like(c)
    else:
        m = optimizer_step.m
        v = optimizer_step.v

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * gradient_matrix

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (gradient_matrix ** 2)

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1 ** (step + 1))

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2 ** (step + 1))

    # Update parameters
    c = c - learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)

    # Store moment estimates for next iteration
    optimizer_step.m = m
    optimizer_step.v = v

    return c
def gradient(c, eta=0.001):
    """
    Computes the gradient of the loss with respect to parameters.

    Parameters:
    loss (torch.Tensor): The loss value.
    c (torch.Tensor): Parameters with respect to which the gradient is computed.

    Returns:
    torch.Tensor: Gradient of the loss with respect to parameters.
    """
    loss_function=functions.loss(c)  
    loss=torch.tensor(loss_function, requires_grad=True)
    grad=[]
    for c_i in c:
        phase =np.random.rand()*2*np.pi
        eta=eta* torch.exp(torch.tensor(1j*phase))
        # so in a sense we create a stochastic gradient descent
        loss_eta=functions.loss((c+eta*c_i)/ (torch.norm(c+ eta*c_i)))
        delta_loss=loss_eta-loss_function
        c_i_grad=delta_loss/eta
        grad.append(c_i_grad)
    c_grad=torch.tensor(grad, requires_grad=True)
    return c_grad



def main():
    return 0

if __name__ == '__main__':
    print(gradient(torch.tensor(functions.create_random(), requires_grad=True)))
    sys.exit(main())



