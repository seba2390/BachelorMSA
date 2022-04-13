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
import numpy as np
from tqdm import tqdm
from plotter import plot_count_histogram
from matplotlib import pyplot as plt



# ------- Set up our QUBO dictionary -------

# Initialize our Q matrix
Q = defaultdict(int)


# ------- Loading MSA QUBO model -------
my_strings         = np.array(["AGC","G","C"])
my_penalties       = np.array([1,1,1])*6
my_msa             = MultipleSequenceAlignment(strings = my_strings, penalties = my_penalties) 
msa_Q,h,d          = my_msa.QUBO_model

good_sol1          = np.array([1,0,0,0,1,0,0,0,1,0,1,0,1,0,0])
good_sol2          = np.array([1,0,0,0,1,0,0,0,1,0,0,1,1,0,0])
good_sol3          = np.array([1,0,0,0,1,0,0,0,1,0,0,1,0,1,0])
good_sol4          = np.array([1,0,0,0,1,0,0,0,1,0,0,1,0,0,1])
good_sol5          = np.array([1,0,0,0,1,0,0,0,1,1,0,0,1,0,0])
good_sol6          = np.array([1,0,0,0,1,0,0,0,1,1,0,0,0,1,0])
good_sol7          = np.array([1,0,0,0,1,0,0,0,1,1,0,0,0,0,1])

valid_solutions    = [my_msa.initial_bitstring,good_sol1,good_sol2,good_sol3,good_sol4,good_sol5,good_sol6,good_sol7]



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
nr_shots = 3

count_dict = {}
for i in tqdm(range(nr_shots)):
    # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q,
                                chain_strength=chainstrength,
                                num_reads=numruns,
                                label='Test')
    best_state = ""
    for val in list(response.first.sample.values()):
        best_state += str(val)
    if best_state not in list(count_dict.keys()):
        count_dict[best_state] = 1
    else:
        count_dict[best_state] += 1

plot_count_histogram(count_dict, valid_solutions)