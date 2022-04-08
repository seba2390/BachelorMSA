import numpy as np

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import Aer

from scipy.optimize import minimize
from MSA_class import MultipleSequenceAlignment



import matplotlib.pyplot as plt

plt.rc("font", family=["Helvetica", "Arial"])
plt.rc("text", usetex=True)
plt.rc("xtick", top=True, direction="in")
plt.rc("ytick", right=True, direction="in")

class QiskitSimulation:
    def __init__(self, initial_MSA, theta_0, backend_name="aer simulator", normalize_weights=True, shots = 512):

        self.initial_MSA     = initial_MSA
        self.current_theta   = theta_0
        self.shots           = shots
        self.current_backend = self.get_Aer_backend(backend_name)
        self.current_circuit = self.set_current_circuit(self.initial_MSA,self.current_theta,normalize_weights)
        self.current_counts  = None

    def compute_Q_edges(self, Q: np.ndarray) -> np.ndarray:
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

    def get_Aer_backend(self, name: str):
        """Function for retrieving specified Aer backend
        Possible backends include:
        
        "aer_simulator", "aer_simulator_statevector","aer_simulator_density_matrix",
        "aer_simulator_stabilizer","aer_simulator_matrix_product_state",
        "aer_simulator_extended_stabilizer","aer_simulator_unitary",
        "aer_simulator_superop","qasm_simulator,statevector_simulator",
        "unitary_simulator","pulse_simulator"

        Parameters:
        -----------
            name: string - name of chosen Aer qiskit backend

        Returns:
        --------
            my_backend: qiskit.providers.aer.backends - chosen backend
        """
        return Aer.get_backend(name)
    
    def print_backend_names(self):
        for backend in Aer.backends():
            print(backend.name())

    def set_current_circuit(self, my_MSA, theta, normalize_weights=True):
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

        edges = self.compute_Q_edges(Q)

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

    def string_to_arr(self, string: str) -> np.ndarray:
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

    def compute_expectation(self, my_MSA: np.ndarray, counts, normalize_weights = True) -> float:
        
        """
        Computes expectation value based on measurement results

        """
        Q, h, d = my_MSA.QUBO_model[0], my_MSA.QUBO_model[1], my_MSA.QUBO_model[2]

        if normalize_weights:
            Q *= 1./np.max(np.array([np.max(Q),np.max(h)]))
            h *= 1./np.max(np.array([np.max(Q),np.max(h)]))

        sum_count, avg = 0, 0
        for bitstring, count in counts.items():
            state = self.string_to_arr(bitstring)
            score = ((state.T @ (Q @ state)) + (h.T @ state) + d)[0][0]
            avg  += score * count
            sum_count += count
        return avg/sum_count

    def execute_circuit(self,theta):
        self.current_circuit = self.set_current_circuit(self.initial_MSA,theta)
        self.current_counts  = self.current_backend.run(self.current_circuit,shots = self.shots,seed_simulator=10).result().get_counts()

        
        return self.compute_expectation(self.initial_MSA,self.current_counts) 
        

    def show_circuit(self):
        if self.current_cirquit != None:
            print(self.current_cirquit)
        else:
            print("Currently no circuit is set")

    def print_solution(self,idx):
        """ Function for getting the idx'th most probable solution"""

        ## Getting values 
        initial_states = np.array(list(self.current_counts.keys()))
        initial_counts = np.array(list(self.current_counts.values()))

        sorting_idxs   = np.flip(np.argsort(initial_counts)) # Flipping for sorting biggest,...,smallest
        initial_states = initial_states[sorting_idxs]
        print(initial_states[:4])
        print("#"*53)
        if idx == 0: string = f"#### Most probale state according to simulation ####"
        else: string = f"#### {idx}'th most probale state according to simulation ####"
        print(string)
        lines = "-"*int((len(string)-len(initial_states[idx]))/2-1)
        print(lines+"|"+initial_states[idx]+"|"+lines)
        print("#"*len(string))
        print(f"with corresponding MSA:")
        print(self.initial_MSA.bit_state_2_matrix(self.string_to_arr(initial_states[idx]).flatten()))
        print("#"*len(string))

    
    def plot_count_histogram(self, counts, solutions, top_number = 55):
        
        ## Getting values 
        initial_states = np.array(list(counts.keys()))
        initial_counts = np.array(list(counts.values()))

        ## Sorting
        initial_counts = np.array([count/np.sum(initial_counts) for count in initial_counts])
        #nr_ones = [np.sum(self.string_to_arr(initial_states[i]).flatten()) for i in range(len(initial_states))]
        #sort_idx = np.argsort(nr_ones)                  ## Sorting after number of ones in states : low  -> high
        sort_idx = np.flip(np.argsort(initial_counts))  ## Sorting after occurrence               : high -> low

        sorted_states = initial_states[sort_idx]
        sorted_counts = initial_counts[sort_idx]

        if top_number < len(sorted_counts):
            sorted_states = sorted_states[:top_number]
            sorted_counts = sorted_counts[:top_number]


        ## Setting idx for states if present in solutions
        good_indexes = []
        for solution in solutions:
            for idx, state in enumerate([self.string_to_arr(sorted_states[i]).flatten() for i in range(len(sorted_states))]):
                equal = True
                for int_idx, integer in enumerate(state.astype(np.float64)):
                    if integer != solution[int_idx]:
                        equal = False
                if equal: good_indexes.append(idx)

        ## Plotting
        fig, ax = plt.subplots(1,1,figsize=(25,6))

        xs = np.arange(0,len(sorted_states))
        x_labels = [r"$|$"+state+r"$\rangle$" for state in sorted_states]
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation = 90,size=15)
        ax.set_title(f"{len(sorted_counts)} most probable states",size=23)
        bar = ax.bar(sorted_states,sorted_counts,align = "center",color=["tab:red" if i in good_indexes else "tab:blue" for i in range(len(xs))],label="Blue is invalid solutions")

        for idx, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{sorted_counts[idx]:.3f}', ha='center', va='bottom')

        ax.set_ylabel("Probability",size=18)
        ax.legend()
        plt.show()