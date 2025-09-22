import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sklearn

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
def gradient(loss, c):
    """
    Computes the gradient of the loss with respect to parameters.

    Parameters:
    loss (torch.Tensor): The loss value.
    c (torch.Tensor): Parameters with respect to which the gradient is computed.

    Returns:
    torch.Tensor: Gradient of the loss with respect to parameters.
    """
    loss.backward()
    return c.grad
def loss(c):

if __name__=="__main__":
    print(0)



