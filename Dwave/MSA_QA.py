# Copyright 2019 D-Wave Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------ Import necessary packages ----
from collections import defaultdict
from MSA_class import MultipleSequenceAlignment

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt



# ------- Set up our QUBO dictionary -------

# Initialize our Q matrix
Q = defaultdict(int)


# ------- Loading MSA QUBO model -------
my_strings   = np.array(["AG","G"])
my_penalties = np.array([1,1,1])*3
my_msa       = MultipleSequenceAlignment(strings = my_strings, penalties = my_penalties) 
msa_Q,h,d    = my_msa.QUBO_model


# Update Q dictionary
for i in range(msa_Q.shape[0]):
    for j in range(msa_Q.shape[1]):
        if i == j:
            Q[(i,i)]+= h[i]       ## Linear terms from h
            Q[(j,i)]+= msa_Q[j,i] ## Diagonal quadratic terms from Q
        else:
            Q[(i,j)]+= msa_Q[i,j] ## Off-diagonal quadratic terms from Q

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 8
numruns = 10

# Run the QUBO on the solver from your config file
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - Maximum Cut')
# ------- Print results to user -------
print('-' * 13)
print("Initial MSA:")
print('-' * 13)
print(f"{my_msa.initial_MSA}\n")
print('-' * 35)
print("with corresponding binary encoding:")
print('-' * 35)
init_state = "|"
for nr in my_msa.initial_bitstring:
    init_state += str(int(nr))
init_state += ">"
print(f"{init_state}\n")
print('-' * 17)
print("Using penalities:")
print('-' * 17)
print(f"p1,p2,p3 = {my_penalties[0],my_penalties[1],my_penalties[2]}\n")
print("#"*8+" ANNEALING RESULTS "+"#"*8)
print('-' * 35)
print('{:>11s}{:>19s}'.format('State:','Energy:'))
print('-' * 35)
for sample, E in response.data(fields=['sample','energy']):
    state = "|"
    for k,v in sample.items():
        state += str(v)
    state += ">"
    print('{:>12s}{:>16.5s}'.format(str(state),str(E)))
print('-' * 55)
print("With lowest energy state corresponding to alignement:")
print('-' * 55)
best_state = []
for val in list(response.first.sample.values()):
    best_state.append(val)
best_alignment = my_msa.bit_state_2_matrix(np.array(best_state))
print(best_alignment)
