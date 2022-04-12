import numpy as np
from itertools import permutations
from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit import Aer
import networkx as nx

from scipy.optimize import minimize



def compute_expectation(counts , Q , h , d , weights, penalties , msa):
    
    """
    Computes expectation value based on measurement results

    """
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        state = string_to_arr(bitstring)
        #score = ((state.T @ (Q @ state)) + (h.T @ state) + d)[0][0]
        score = calculate_cost(weights , msa , state.T[0] , penalties)
        avg += score * count
        sum_count += count
    return avg/sum_count

def create_qaoa_circ(initial_MSA , edges , penalties , theta):
    
    """
    Creates a parametrized qaoa circuit

    """
    letters_in_mat = [initial_MSA[i][j] for i in range(initial_MSA.shape[0]) 
                      for j in range(initial_MSA.shape[1]) if initial_MSA[i][j] != "_"]
    
    nqubits = int(len(letters_in_mat) *  initial_MSA.shape[1])  # Number of qubits = number of nodes in graph 
    p       = len(theta)//2                                     # Number of alternating unitaries
    qc      = QuantumCircuit(nqubits)                           # Initializing Q circuit w. nqubits nr. of qbits

    beta  = theta[:p]                # Beta opt param for mixing unitaries as first p vals.
    gamma = theta[p:]                # Gama opt param for cost unitaries as last p vals.
    
    p1, p2, p3 = penalties[0], penalties[1], penalties[2]

    
    # Initial_state: Hadamark gate on each qbit
    for i in range(0, nqubits):
        qc.h(i)
    
    # Cost unitary:
    for irep in range(0, p):        
        for i, j, w_ij in edges:
            if i!= j:
                qc.rzz(2 * gamma[irep] * w_ij , i , j)
            else:
                qc.rz(2 * gamma[irep] * w_ij  , i  )   # Diagonal in Q matrix
                qc.rz(2 * gamma[irep] * (-1 * p1) , i) # Elements of h vector

    # mixer unitary: X rotation on each qbit      
    for irep in range(0, p): 
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
            
    qc.measure_all()        
    return qc

# Finally we write a function that executes the circuit on the chosen backend
def get_expectation(initial_MSA: np.ndarray, Q: np.ndarray , h: np.ndarray , 
                    d: float , edges: list , p: int , weights, penalties, shots=512):
    
    """
    Runs parametrized circuit
    
    """
    #backend = StatevectorSimulator(precision="double")
    backend = Aer.get_backend('qasm_simulator')
    #backend.shots = shots
    
    def execute_circ(theta):
        
        qc = create_qaoa_circ(initial_MSA , edges , penalties , theta)
        #qc = qc.decompose()
        #qc = transpile(qc,optimization_level=3)
        counts = backend.run(qc).result().get_counts()
        return compute_expectation(counts , Q , h , d , weights,penalties, initial_MSA) ## Returns the expectation of graph, given counts
    
    return execute_circ

def columnwise_norm(mat):
    for i in range(mat.shape[1]):
        mat[:,i] *= 1 / np.sum(mat[:,i])

def rowwise_norm(mat):
    for i in range(mat.shape[0]):
        mat[i] *= 1 / np.sum(mat[i])