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

def find_coefficients(c, angles, target):
    alphas=[0]
    # the first angle is initialized to be 0 and we will figure out the remaining angles
    # for each angle, we first form a superposition of the state with psi1
    for i in range(1,32):
        # we first form the state 1/sqrt(2)([1 ... 0] + [0 .. c_i * e^alpha_i .. 0]])
        state1=np.zeros(32)
        state1[0]=1/np.sqrt(2)* c[0]* np.exp(1j*angles[0])
        state2=np.zeros(32)
        state2[i]=1/np.sqrt(2)* c[i]* np.exp(1j*angles[i])
        state=state1+state2
        # now we do the SWAP on the target and this constructed state


        qc=QuantumCircuit(11)   
        qc.initialize(state, [0,1,2,3,4], normalize=True)
        qc.initialize(target, [5,6,7,8,9], normalize=True)
        qc.initialize([1,0],10, normalize=True)
        qc.add_register(ClassicalRegister(1, 'c'))
        qc.h(10)
        qc.append(CSwapGate(), [10, 0, 5])
        qc.append(CSwapGate(), [10, 1, 6])
        qc.append(CSwapGate(), [10, 2, 7])
        qc.append(CSwapGate(), [10, 3, 8])
        qc.append(CSwapGate(), [10, 4, 9])
        qc.h(10)
        simulator = AerSimulator(method='statevector')
        qc.measure(10,0)
        job = simulator.run(qc, shots=100)  # Run 100 times
        result = job.result()
        counts = result.get_counts(qc)
        num_zeros = counts.get('0', 0)
        aa= np.sqrt(np.absolute(2*(num_zeros / 100)-1.0))
        # we can now extract the angle alpha_i from the fidelity value
        quantity=aa - 1 - (c[i]/c[0])**2
        quantity=quantity/(2*c[i]/c[0])
        if quantity>1:  
            quantity=1
        if quantity<-1:
            quantity=-1
        alpha_i=np.arccos(quantity)
        alphas.append(alpha_i)
    return alphas


if __name__ == '__main__':
    target=functions.create_random()*np.exp(1j*np.random.rand(32)*2*np.pi)
    true_alphas=find_coefficients(torch.tensor(setup.initialize_c(target,setup.basis(5))), torch.tensor(functions.create_init()),
                                        torch.tensor(target))
    
    sys.exit(0)