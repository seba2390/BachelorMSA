import numpy as np

from qiskit import QuantumCircuit
from qiskit import Aer

from MSA_class import MultipleSequenceAlignment

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm

from Qiskit_class import QiskitSimulation


def cost_eval(my_sim, theta):
    qc = my_sim.set_current_circuit(my_sim.initial_MSA, theta)
    my_sim.current_counts = my_sim.current_backend.run(qc,seed_simulator=10).result().get_counts()
    return my_sim.compute_expectation(my_sim.initial_MSA, my_sim.current_counts) 


my_strings   = np.array(["AG","G"])
my_penalties = np.array([1,1,1])*2
my_msa       = MultipleSequenceAlignment(strings = my_strings, penalties = my_penalties, normalize_weights=True) 

p = 2
theta_0 = np.ones(p)
my_simulation = QiskitSimulation(initial_MSA=my_msa, theta_0=theta_0, backend_name="qasm_simulator", shots=10000)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(10,10))

# Make data.
X = np.linspace(0, 4*np.pi, 300)
Y = np.linspace(0, 4*np.pi, 300)
X, Y = np.meshgrid(X, Y)
final_Z = []
for col in tqdm(range(Y.shape[1])):
    row_Z = []
    for row in range(X.shape[0]):
        cur_x, cur_y = X[row][col], Y[row][col]
        row_Z.append(cost_eval(my_simulation,[cur_x,cur_y]))
    final_Z.append(row_Z)
Z = np.array(final_Z)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

ax.zaxis.set_ticks(np.linspace(np.min(Z),np.max(Z),10))

plt.show()