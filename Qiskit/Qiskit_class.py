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
    def __init__(self, initial_MSA, theta_0, backend_name="aer simulator", shots = 512):

        self.initial_MSA     = initial_MSA
        self.current_theta   = theta_0
        self.shots           = shots
        self.current_backend = self.get_Aer_backend(backend_name)
        self.current_circuit = self.set_current_circuit(self.initial_MSA,self.current_theta)
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

    def inverse_dict_states(self, count_state_dict):
        """ Qiskit takes the right-most qubit as the zeroth. This
            functions takes a dictionary of {state:count} and inverses 
            the states s.t. 0001 -> 1000 
        
        Paramaters:
        -----------
            count_state_dict: dictionary, e.g.: {state1:count1, state2:counts2,...}

        Returns:
        --------
            reversed_state_dict: dictionary, e.g.: {reversed(state1):count1, reversed(state2):counts2,...}
        """
        reversed_state_dict = {}
        for idx, key in enumerate(list(count_state_dict.keys())):
            reversed_state_dict[key[::-1]] = list(count_state_dict.values())[idx]
        return reversed_state_dict


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
    

    def set_current_circuit(self, my_MSA, theta):
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

        """
        Q, h, d    = my_MSA.QUBO_model[0], my_MSA.QUBO_model[1], my_MSA.QUBO_model[2]

        """
        J, g, c = my_MSA.Ising_model[0], my_MSA.Ising_model[1], my_MSA.Ising_model[2]

        # Initial_state: Hadamark gate on each qbit
        for i in range(0, nqubits):
            qc.h(i)
        
        
        edges = self.compute_Q_edges(J)
        for irep in range(0, p):

            # Cost unitary:        
            for i, j, w_ij in edges:
                if i!= j:
                    qc.rzz(2 * gamma[irep] * w_ij , i , j)  # Off-Diagonal in Q matrix
                else:
                    qc.rz(2 * gamma[irep] * w_ij, i)        # Diagonal in Q matrix
            for i, g_i in enumerate(g):
                qc.rz(2 * gamma[irep] * g_i, i)             # Elements of h vector
            

            """
            # Cost unitary:        
            for i in range(J.shape[0]):
                for j in range(J.shape[1]):
                    if i == j:
                        if J[i][j] != 0:
                            qc.rz(2 * gamma[irep] * J[j][i] , i)
                            qc.rz(2 * gamma[irep] * g[i] , i)
                    else:
                        if J[i][j] != 0:
                            qc.rzz(2 * gamma[irep] * J[i][j] , i , j) # Diagonal in Q matrix
            """

            # Mixer unitary: X rotation on each qbit     
            for i in range(0, nqubits):
                qc.rx(2 * beta[irep], i)


                
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

    def compute_expectation(self, my_MSA: np.ndarray, counts) -> float:
        
        """
        Computes expectation value based on measurement results

        """
        
        J, g, c    = my_MSA.Ising_model[0], my_MSA.Ising_model[1], my_MSA.Ising_model[2]

        sum_count, avg = 0, 0
        for bitstring, count in counts.items():
            state = self.string_to_arr(bitstring)
            score = ((state.T @ (J @ state)) + (g.T @ state))[0][0]
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
        """ Function for getting and printing the idx'th most 
            probable state foind in self.current counts:
        
        Parameters:
        -----------
            idx: int
        """

        count_state_dict = self.inverse_dict_states(self.current_counts)

        ## Getting values 
        initial_states = np.array(list(count_state_dict.keys()))
        initial_counts = np.array(list(count_state_dict.values()))

        sorting_idxs   = np.flip(np.argsort(initial_counts, kind = "heapsort")) # Flipping for sorting biggest,...,smallest
        initial_states = initial_states[sorting_idxs]
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
        """ Function for plotting a historgram of the states and their corresponding
            probability.
            
        Parameters:
        -----------
            counts    : dictionary, e.g.: {state1:count1, state2:counts2,...}
            solutions : list of solutions, e.g.: [valid_state1,valid_state2,...]
            top_number: int - max number of most probable states displayed
        
        """
        
        #---------- Getting values ----------#
        initial_states = np.array(list(counts.keys()))
        initial_counts = np.array(list(counts.values()))

        #---------- Sorting ----------#
        initial_counts = np.array([count/np.sum(initial_counts) for count in initial_counts])

        #---------- Sorting after number of ones in states : low  -> high ----------#
        #nr_ones = [np.sum(self.string_to_arr(initial_states[i]).flatten()) for i in range(len(initial_states))]
        #sort_idx = np.argsort(nr_ones,kind="heapsort")  

        #---------- Sorting after occurrence               : high -> low -----------#            
        sort_idx = np.flip(np.argsort(initial_counts,kind="heapsort"))  

        sorted_states = initial_states[sort_idx]
        sorted_counts = initial_counts[sort_idx]

        if top_number < len(sorted_counts):
            sorted_states = sorted_states[:top_number]
            sorted_counts = sorted_counts[:top_number]


        #---------- Setting idx for states if present in solutions -----------#
        good_indexes = []
        for solution in solutions:
            for idx, state in enumerate([self.string_to_arr(sorted_states[i]).flatten() for i in range(len(sorted_states))]):
                equal = True
                for int_idx, integer in enumerate(state.astype(np.float64)):
                    if integer != solution[int_idx]:
                        equal = False
                if equal: good_indexes.append(idx)

        #---------- Plotting -----------#
        fig, ax = plt.subplots(1,1,figsize=(25,6))

        xs = np.arange(0,len(sorted_states))
        x_labels = [r"$|$"+state+r"$\rangle$" for state in sorted_states]
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation = 90,size=15)
        ax.set_title(f"{len(sorted_counts)} most probable states (Red corresponds to valid MSA's)",size=23)
        bar = ax.bar(sorted_states,sorted_counts,align = "center",color=["tab:red" if i in good_indexes else "tab:blue" for i in range(len(xs))],label="Blue is invalid solutions")

        for idx, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{sorted_counts[idx]:.3f}', ha='center', va='bottom')

        ax.set_ylabel("Probability",size=18)
        fig.subplots_adjust(bottom=0.2) ## Increasing space below fig (in case of large states)
        plt.show()