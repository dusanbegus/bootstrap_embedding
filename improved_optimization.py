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

# we will now try to go through the alphas one by one and compute the gradient with respect to each alpha
def optimizer_step_improved(angle, eta, gradient):
    # Performs a single optimization step using the computed gradient.
    # Parameters:       
    # angle (torch.Tensor): Current parameter to be optimized.
    # eta (float): Learning rate for the optimization step. 
    angle= angle- eta * gradient
    return angle
def gradient_improved(c, angles, index, target, eta=0.01):  
    # angle is the angle we are focusing on, and index marks its position in the angles list
    # Computes the gradient of the loss with respect to a single parameter.
    # Parameters:
    # loss (torch.Tensor): The loss value.
    # c (torch.Tensor): Parameters with respect to which the gradient is computed.  
    loss_function=functions.loss(c, angles, target_state=target)  
    # we now construct a perturbation only for the index we are interested in
    perturbation=torch.zeros(len(angles))
    perturbation[index]=eta
    loss_alpha=functions.loss(c, angles+perturbation, target_state=target)
    delta_loss=loss_alpha-loss_function
    alpha_grad=delta_loss/eta
    return alpha_grad
def gradient_descent_improved(c, angles, target, steps=1000, eta=0.01):
    for i, angle in enumerate(angles):
        for step in range(steps//len(angles)):
            angles_grad = gradient_improved(c,angles, i, target, eta)
            angles[i]= optimizer_step_improved(angles[i], eta, angles_grad)
            
    return c, angles

if __name__ == '__main__':
    # in the main loop we go through the angles one by one and do the gradient descent for each angle and then we move on to the next angle
    target=functions.create_random()*np.exp(1j*np.random.rand(32)*2*np.pi)
    initial_angles=torch.tensor(functions.create_init())
    initial_c=torch.tensor(setup.initialize_c(target,setup.basis(5)))
    a=gradient_descent_improved(initial_c, initial_angles.clone(),
                                        torch.tensor(target),
                                        steps=100, eta=0.1)
    print("target state: ", target)
    print("initial state", torch.tensor(initial_c*np.exp(1j*initial_angles.detach().numpy())))
    print("initial norm of inner product with target state", np.linalg.norm(np.inner(initial_c.numpy()*np.exp(1j*initial_angles.detach().numpy()),target)))
    print("initial fidelity with target state: ", 1-functions.loss(initial_c, initial_angles, target_state=target))
    print("final state: ", a[0]*np.exp(1j*a[1].detach().numpy()))
    print("norm of inner product with target state in the end", np.linalg.norm(np.inner(a[0]*np.exp(1j*a[1].detach().numpy()),target)))
    print("final fidelity with target state: ", 1-functions.loss(a[0], a[1], target_state=target))
    sys.exit(0)
   
    
