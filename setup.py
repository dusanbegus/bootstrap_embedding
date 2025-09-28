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
def basis(n):
    # n is the number of qubits
    basis_states=[]
    for i in range(2**n):
        state=[0 for _ in range(2**n)]
        state[i]=1
        basis_states.append(state)
    return basis_states
def initialize_c(state, basis):
    if torch.is_tensor(state):
        state=state.detach().numpy()
    if torch.is_tensor(basis):
        basis=basis.detach().numpy()
    cs=[]
    for vector in basis:
        qc=QuantumCircuit(11)
        
        qc.initialize(state, [0,1,2,3,4], normalize=True)
        qc.initialize(vector, [5,6,7,8,9], normalize=True)  
        
        # now we will compute the fidelity between these two states
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
        
        job = simulator.run(qc, shots=500)  # Run 100 times
        result = job.result()
        counts = result.get_counts(qc)
        num_zeros = counts.get('0', 0)
        cs.append(np.sqrt(num_zeros / 500))
    return cs
if __name__ == "__main__":
    print("Testing basis function...")
    b = basis(5)
    print(f"Generated {len(b)} basis states for 5 qubits.")
    print("Testing initialize_c function...")
    init_state = [1/np.sqrt(32) for _ in range(32)]
    c_values = initialize_c(init_state, b)
    print(f"Computed c values: {c_values}")
    sys.exit(0)
        
    


