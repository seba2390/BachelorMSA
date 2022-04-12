import numpy as np

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import Aer

from scipy.optimize import minimize
from MSA import MultipleSequenceAlignment

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def compute_Q_edges(Q: np.ndarray) -> np.ndarray:
    """ Function for computing edges from Q matrix in QUBO
    
    Parameters:
    -----------
        Q: np.ndarray - 2D numpy array containing weights in Q matrix from QUBO
    
    Returns:
    --------
       edges: np.ndarray - 1D numpy array of triplets np.array([(i_1,j_1,w_i1j1),(i_2,j_2,w_i2j2),...])

    """

    edges = [] 
    for idx1 in range(Q.shape[1]):
        for idx2 in range(Q.shape[0]):
            if Q[idx1][idx2] != 0:
                edges.append((idx1,idx2,Q[idx1][idx2]))
    return edges

def create_qaoa_circ(my_MSA, theta, normalize_weights=True):
    
    """
    Creates a parametrized qaoa circuit

    """
    letters_in_mat = [my_MSA.initial_MSA[i][j] for i in range(my_MSA.initial_MSA.shape[0]) 
                      for j in range(my_MSA.initial_MSA.shape[1]) if my_MSA.initial_MSA[i][j] != "_"]
    
    nqubits = int(len(letters_in_mat) *  my_MSA.initial_MSA.shape[1])  # Number of qubits = number of nodes in graph 
    p       = len(theta)//2                                            # Number of alternating unitaries
    qc      = QuantumCircuit(nqubits)                                  # Initializing Q circuit w. nqubits nr. of qbits

    beta  = theta[:p]                # Beta opt param for mixing unitaries as first p vals.
    gamma = theta[p:]                # Gama opt param for cost unitaries as last p vals.
    
    p1, p2, p3 = my_MSA.penalties[0],  my_MSA.penalties[1],  my_MSA.penalties[2]

    Q, h, d    = my_MSA.QUBO_model[0], my_MSA.QUBO_model[1], my_MSA.QUBO_model[2]

    if normalize_weights:
        Q *= 1./np.max(np.array([np.max(Q),np.max(h)]))
        h *= 1./np.max(np.array([np.max(Q),np.max(h)]))

    edges = compute_Q_edges(Q)

    # Initial_state: Hadamark gate on each qbit
    for i in range(0, nqubits):
        qc.h(i)
    
    # Cost unitary:
    for irep in range(0, p):        
        for i, j, w_ij in edges:
            if i!= j:
                qc.rzz(2 * gamma[irep] * w_ij , i , j)
            else:
                qc.rz(2 * gamma[irep] * w_ij      , i) # Diagonal in Q matrix
                qc.rz(2 * gamma[irep] * h[i] * p1 , i) # Elements of h vector

    # Mixer unitary: X rotation on each qbit      
    for irep in range(0, p): 
        for i in range(0, nqubits):
            qc.rx(1 * beta[irep], i)
            
    qc.measure_all()        
    return qc

def string_to_arr(string: str) -> np.ndarray:
    """Funtion for transforming given string of integers
       to corresponding array of ints

    Parameters:
    -----------
        string: str - a string of ints, e.g.: '001100'   

    Returns:
    --------
        arr: np.ndarray - numpy array of ints, e.g.: np.array([0,0,1,1,0,0])
    """

    arr = []
    for str in string: arr.append(int(str))
    return np.array(arr).reshape((len(np.array(arr)),1))
    

def compute_expectation(my_MSA: np.ndarray, counts) -> float:
    
    """
    Computes expectation value based on measurement results

    """
    Q, h, d = my_MSA.QUBO_model[0], my_MSA.QUBO_model[1], my_MSA.QUBO_model[2]
    sum_count, avg = 0, 0
    for bitstring, count in counts.items():
        state = string_to_arr(bitstring)
        score = ((state.T @ (Q @ state)) + (h.T @ state) + d)[0][0]
        avg  += score * count
        sum_count += count
    return avg/sum_count


def cost_eval(my_MSA, theta, backend):
    qc = create_qaoa_circ(my_MSA, theta, normalize_weights=True)
    #qc = qc.decompose()
    #qc = transpile(qc,optimization_level=3)
    counts = backend.run(qc,seed_simulator=10).result().get_counts()
    return compute_expectation(my_MSA,counts) ## Returns the expectation of graph, given counts


my_strings   = np.array(["AG","G"])
my_penalties = np.array([1,1,1])
my_msa       = MultipleSequenceAlignment(strings = my_strings, penalties = my_penalties) 


shots   = 512
theta_0 = np.array([1,1,1,1]) 
backend = Aer.get_backend('aer_simulator')
backend.shots = shots

print(cost_eval(my_msa,theta_0,backend))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.linspace(0, 2*np.pi, 100)
Y = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(X, Y)
final_Z = []
for col in tqdm(range(Y.shape[1])):
    row_Z = []
    for row in range(X.shape[0]):
        cur_x, cur_y = X[row][col], Y[row][col]
        row_Z.append(cost_eval(my_msa,[cur_x,cur_y],backend))
    final_Z.append(row_Z)
Z = np.array(final_Z)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()