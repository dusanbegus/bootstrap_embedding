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
import setup

def gradient_descent(c, angles, target, steps=1000, eta=0.01):
    """
    Performs gradient descent optimization on the parameters.

    Parameters:
    c (torch.Tensor): Initial parameters to be optimized.
    steps (int): Number of optimization steps (iterations).

    Returns:
    torch.Tensor: Optimized parameters after the specified number of steps.
    """
    previous=1.0
    original_loss=functions.loss(c, angles, target_state=target)
    minimal_loss=original_loss
    angles_minimal=angles
    for step in range(steps):
        angles_grad = optimizer.gradient(c,angles, target, eta)
        angles_f = optimizer.optimizer_step(angles, eta, angles_grad)
        
        if step % 1 == 0:

            loss_value=functions.loss(c, angles_f, target_state=target)
            if loss_value<minimal_loss:
                minimal_loss=loss_value
                angles_minimal=angles_f
            if loss_value<previous*0.5:
                eta=eta*0.3
                previous=loss_value
            else:
                eta=eta*1.2
                previous=loss_value
            print(f"Step {step}, Loss: {loss_value}, Learning Rate: {eta}")
    return c, angles_minimal, minimal_loss
if __name__ == '__main__':
    target=functions.create_random()*np.exp(1j*np.random.rand(32))
    a=gradient_descent(torch.tensor(setup.initialize_c(target,setup.basis(5))), torch.tensor(functions.create_init()),
                                        torch.tensor(target),
                                        steps=50)
    # we now compute the fidelity by subtracting the loss from 1
    fidelity = 1 - a[2] 
    print("target state:  ", target)
    print("optimized state:  ", a[0]*np.exp(1j*a[1].detach().numpy()))
    # we want to compute the norm of the difference
    print("norm of difference", np.linalg.norm(a[0]*np.exp(1j*a[1].detach().numpy())-target))
    print(f"Fidelity with target state: {fidelity}")    
    
    sys.exit(0)