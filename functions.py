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
print("Imports completed, starting main code...")

def create_init():
    l=[]
    for i in range(32):
        l.append(1/np.sqrt(32))
    return l    
def create_random():
    state=np.random.rand(32)
    state=state/np.linalg.norm(state)
    return state.tolist()   
print('okay, functions defined')
def loss(c, angles, target_state):
    if torch.is_tensor(c):
        c=c.detach().numpy()
    print('ok')
    if torch.is_tensor(angles):
        angles=angles.detach().numpy()
    print('ok1')
    # we wil have a list of 2^5 parameters initially and we will perform optimization over them
    phases=[np.exp(1j*angle) for angle in angles]
    state=np.array(c)*np.array(phases)
    print('ok2')
    print(f"state shape: {state.shape}")
    print(f"state dtype: {state.dtype}")
    print(f"target_state shape: {np.array(target_state).shape}")
    print(f"target_state dtype: {np.array(target_state).dtype}")
    print(f"state sum: {np.sum(np.abs(state)**2)}")  # Should be ~1 for normalized

    # phases contain the phases of the states
    qc=QuantumCircuit(11)
    
    qc.initialize(state, [0,1,2,3,4], normalize=True)
    qc.initialize(target_state, [5,6,7,8,9], normalize=True)  
    print('ok3')
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
    print('ok4')
    job = simulator.run(qc, shots=100)  # Run 100 times
    result = job.result()
    counts = result.get_counts(qc)
    num_zeros = counts.get('0', 0)
    fidelity = num_zeros / 100
    
    return 1-fidelity

print('loss function defined')
if __name__ == "__main__":
    print("Testing create_init()...")
    init1 = create_init()
    print("First create_init() worked")
    
    init1_list = list(init1)
    print("First list() conversion worked")
    
    init2 = list(create_init())
    print("Second create_init() worked")
    
    init3 = list(create_init())
    print("Third create_init() worked")
    
    print("Testing loss function...")
    result = loss(init1_list, init2, init3)
    print("Loss function result:", result)
    sys.exit(0)
    