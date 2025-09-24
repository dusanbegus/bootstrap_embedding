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
import optimizer

def create_init():
    l=[]
    for i in range(32):
        l.append(1/np.sqrt(32))
    return l    

def loss(c, target_state=list(create_init())):
    # we wil have a list of 2^5 parameters initially and we will perform optimization over them
    state=np.array(c)
    qc=QuantumCircuit(11)
    qc.initialize(state, [0,1,2,3,4])
    qc.initialize(target_state, [5,6,7,8,9])  
    # now we will compute the fidelity between these two states
    qc.initialize([1,0],10)
    qc.add_register(ClassicalRegister(1, 'c'))
    qc.h(10)
    qc.append(CSwapGate(), [10, 0, 5])
    qc.append(CSwapGate(), [10, 1, 6])
    qc.append(CSwapGate(), [10, 2, 7])
    qc.append(CSwapGate(), [10, 3, 8])
    qc.append(CSwapGate(), [10, 4, 9])  
    qc.h(10) 
    simulator = AerSimulator()
    qc.measure(10,0)
    job = simulator.run(qc, shots=100)  # Run 100 times
    result = job.result()
    counts = result.get_counts(qc)
    num_zeros = counts.get('0', 0)
    fidelity = num_zeros / 100
    print("Fidelity:", fidelity)


if __name__ == "__main__":
    loss(list(create_init())        )
    sys.exit(0)
    