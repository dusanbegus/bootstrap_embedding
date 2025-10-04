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
import optimization


def tests():
    # we will create some tests to check the gradient descent
    # just like in optimization.py, we initialize a random target state, we measure the amplitudes via SWAP, and then we do the gradient descent on alphas
    # we will check if the fidelity increases after the gradient descent
    # we will plot variance of the distribution of alphas vs improvement in fidelity
    # we will also plot the initial fidelity vs improvement in fidelity
    # we will also plot the initial inner product vs improvement in inner product
    # finally, we will test how changing the optimizer step affects the precision of the process over 5 qubits
    average_fideliy_improvement=0
    average_innerproduct_improvement=0
    initial_fidelities=[]
    fidelity_improvements=[]
    initial_innerproducts=[]
    innerproduct_improvements=[]
    variances=[]
    variance_fidelity_improvements=0
    variance_innerproduct_improvements=0
    for i in range(10):
        # we will keep eta=0.1 and steps=100 fixed and will just initializr the targets randomly
        # let us fix the random seed
        np.random.seed(32)
        target=functions.create_random()*np.exp(1j*np.random.rand(32)*2*np.pi)
        c=setup.initialize_c(target,setup.basis(5))
        angles=torch.tensor(functions.create_init(), requires_grad=True)
        initial_loss=functions.loss(torch.tensor(c), angles, torch.tensor(target))
        initial_fidelities.append(1-initial_loss)
        initial_innerproducts.append(np.linalg.norm(np.inner(c*np.exp(1j*angles.detach().numpy()),target)))
        variances.append(np.var(angles.detach().numpy()))       
        c_opt, angles_opt, final_loss, initial_innerproduct, initial_fidelity=optimization.gradient_descent(torch.tensor(c), angles,
                                            torch.tensor(target),
                                            steps=100, eta=0.1)
        final_fidelity = 1 - final_loss
        final_innerproduct=np.linalg.norm(np.inner(c_opt*np.exp(1j*angles_opt.detach().numpy()),target))
        fidelity_improvements.append(final_fidelity-initial_fidelity)
        innerproduct_improvements.append(final_innerproduct-initial_innerproduct)
        average_fideliy_improvement+=final_fidelity-initial_fidelity
        average_innerproduct_improvement+=final_innerproduct-initial_innerproduct
        variance_fidelity_improvements+=final_fidelity-initial_fidelity
        variance_innerproduct_improvements+=final_innerproduct-initial_innerproduct
    average_fideliy_improvement=average_fideliy_improvement/10
    average_innerproduct_improvement=average_innerproduct_improvement/10
    variance_fidelity_improvements=variance_fidelity_improvements/10
    variance_innerproduct_improvements=variance_innerproduct_improvements/10
    # now we will plot a histogram of fidelity improvements and inner product improvements
    plt.figure()
    plt.hist(fidelity_improvements, bins=10)
    plt.title("Histogram of fidelity improvements over 10 random initializations")
    plt.xlabel("Fidelity improvement")
    plt.ylabel("Frequency")
    plt.savefig("fidelity_improvements_histogram.png")
    plt.figure()
    plt.hist(innerproduct_improvements, bins=10)
    plt.title("Histogram of inner product improvements over 10 random initializations")
    plt.xlabel("Inner product improvement")
    plt.ylabel("Frequency")
    plt.savefig("innerproduct_improvements_histogram.png")
    # now we will plot initial fidelities vs fidelity improvements
    plt.figure()
    plt.scatter(initial_fidelities, fidelity_improvements)
    plt.title("Initial fidelities vs fidelity improvements")
    plt.xlabel("Initial fidelity")
    plt.ylabel("Fidelity improvement")
    plt.savefig("initial_fidelities_vs_fidelity_improvements.png")
    # now we will plot initial inner products vs inner product improvements
    plt.figure()
    plt.scatter(initial_innerproducts, innerproduct_improvements)
    plt.title("Initial inner products vs inner product improvements")
    plt.xlabel("Initial inner product")     
    plt.ylabel("Inner product improvement")
    plt.savefig("initial_innerproducts_vs_innerproduct_improvements.png")
    # now we will plot variances vs fidelity improvements
    plt.figure()
    plt.scatter(variances, fidelity_improvements)
    plt.title("Variances of angles vs fidelity improvements")
    plt.xlabel("Variance of angles")
    plt.ylabel("Fidelity improvement")
    plt.savefig("variances_vs_fidelity_improvements.png")
    # now we will plot variances vs inner product improvements
    plt.figure()
    plt.scatter(variances, innerproduct_improvements)
    plt.title("Variances of angles vs inner product improvements")
    plt.xlabel("Variance of angles")
    plt.ylabel("Inner product improvement")
    plt.savefig("variances_vs_innerproduct_improvements.png")
    print(f"Average fidelity improvement over 10 random initializations: {average_fideliy_improvement}")    
    print(f"Average inner product improvement over 10 random initializations: {average_innerproduct_improvement}")
    print(f"Variance of fidelity improvements over 10 random initializations: {variance_fidelity_improvements}")
    print(f"Variance of inner product improvements over 10 random initializations: {variance_innerproduct_improvements}")  
    return 0
if __name__ == '__main__':
    sys.exit(tests())
