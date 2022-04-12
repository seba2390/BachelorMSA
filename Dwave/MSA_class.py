import numpy as np
from itertools import permutations
from copy import deepcopy

class MultipleSequenceAlignment:
    def __init__(self, strings: np.ndarray, penalties = [1,1,1], alphabet = ["A","C","T","G"], normalize_weights = False):

        self.alphabet           = alphabet

        for string in strings:
            for character in string:
                assert character in self.alphabet, f"The character {character}, is not defined on current alphabet."

        assert type(strings) is np.ndarray, f'Strings given as {type(strings)}, but should be {np.ndarray}'
        assert len(strings) > 1,            f'Only 1 string of letters given - should contain at least 2!'
        assert type(penalties) is list or type(penalties) is np.ndarray, f'Penalties given as {type(penalties)}, but should be {np.ndarray} or {list}'
        assert type(normalize_weights) is bool, f'normalize_weights given as {type(normalize_weights)}, but should be {type(normalize_weights)}'
        
        self.strings             = strings
        self.penalties           = penalties
        self.initial_MSA         = self.MSA_matrix(self.strings)
        self.initial_bitstring   = self.matrix_2_bit_state(self.initial_MSA,self.initial_MSA) ## state in { 0 , 1}^n
        self.initial_isingstring = 1 - 2 * self.initial_bitstring                             ## state in {-1 , 1}^n
        self.score_weights       = self.encode_score_weights()
        self.QUBO_model          = self.encode_QUBO(normalize_weights)  ## self.QUBO_model is given as (Q,h,d)
        self.Ising_model         = self.encode_Ising(normalize_weights) ## self.isng_model is given as (J,g,c)

    def MSA_matrix(self, init_strings: np.ndarray) -> np.ndarray:
        """ Creating a matrix representation of the strings given
        and filling gaps with "_", whilst sort, s.t. longest string
        is on top row of matrix.

        Parameters:
        -----------
            init_strings: np.array of strings, e.g. np.array(["ACCT","AC","AT"])

        Returns:
        --------
            2D numpy array
        """
        ## Asurring longest string on top
        lengths = np.array([len(str) for str in init_strings])
        strings = init_strings[np.flip(np.argsort(lengths, kind="heap"))]
        ## Creating matrix representation
        initial_matrix = np.zeros((len(strings) , np.max(lengths)),dtype=object)
        for row in range(initial_matrix.shape[0]):
            for col in range(len(strings[row])):
                initial_matrix[row][col] = strings[row][col]
        initial_matrix[initial_matrix == 0] = "_"
        return initial_matrix

    def matrix_2_bit_state(self, my_matrix:  np.ndarray, original_matrix:  np.ndarray) -> np.ndarray:
        """
        Maps some given matrix repr. of a MSA "my_matrix" to a corresponding
        bitstring via the column encoding, whilst respecting
        original order in "original_matrix"
        x_(s,n,i) determines whether the n'th letter of the s'th string
        is placed in the i'th column.

        Parameters:
        -----------
            mat: 2D numpy array, e.g. array([['A', 'C', 'C', 'T'],
                                             ['A', 'C', '_', '_'],
                                             ['A', 'T', '_', '_']])
        Returns:
        --------
            numpy array containing bit repr., e.g.: np.array([1,0,0,1...])

        """
        ## Initial definitions
        current_matrix = deepcopy(my_matrix)
        nr_of_rows, nr_of_cols, gap = current_matrix.shape[0], current_matrix.shape[1], "_"
        ## List of nr of letters in i'th row of original matrix
        nr_letters_in_mat = [len(np.where(original_matrix[i] != "_")[0]) for i in range(nr_of_rows)]
        ## Needed nr of registers
        regs = np.array([list(np.zeros(nr_of_cols)) for i in range(np.sum(nr_letters_in_mat))])
        ## List of lists of letters in original matrix rows
        original_letters = np.array([list(original_matrix[i][np.where(original_matrix[i] != gap)[0]]) 
                                    for i in range(nr_of_rows)], dtype=object)
        ## List of lists of letters in current matrix rows
        current_letters = np.array([list(current_matrix[i][np.where(current_matrix[i] != gap)[0]]) 
                                    for i in range(nr_of_rows)], dtype=object)
        for s in range(0 , nr_of_rows):
            for n in range(0 , len(current_letters[s])):
                ## Finding index
                reg_idx = np.where(np.array(original_letters[s]) == current_letters[s][n])[0][0] + int(np.sum(nr_letters_in_mat[:s])) # Original n'th idx
                col_idx = np.where(np.array(current_matrix[s])   == current_letters[s][n])[0][0]                                      # Current i'th idx
                ## Setting reg value
                regs[reg_idx][col_idx] = 1
                ## Changing comparator to "O" (the letter) in case of multiple of same char
                original_letters[s][np.where(np.array(original_letters[s]) == current_letters[s][n])[0][0]] = "O"
                current_matrix[s][ np.where(np.array(current_matrix[s])  == current_letters[s][n])[0][0]] = "O"
        return regs.flatten()
    
    def bit_state_2_matrix(self , bit_string : np.ndarray) -> np.ndarray:
        """
        Maps some given bitstring repr. of a MSA to a corresponding
        matrix via the initial matrix. 

        Parameters:
        -----------
            bit_string: numpy array containing bit repr., e.g.: np.array([1,0,0,1...])
            mat       : 2D numpy array, e.g. array([['A', 'C', 'C', 'T'],
                                                    ['A', 'C', '_', '_'],
                                                    ['A', 'T', '_', '_']])
        Returns:
        --------
            2D numpy array, e.g. array([['A', 'C', 'C', 'T'],
                                        ['A', 'C', '_', '_'],
                                        ['A', 'T', '_', '_']])
        """
        assert type(bit_string) is np.ndarray, f'Bit string given as {type(bit_string)}, but should be {np.ndarray}'

        letters_in_mat = [self.initial_MSA[i][j] for i in range(self.initial_MSA.shape[0]) 
                        for j in range(self.initial_MSA.shape[1]) if self.initial_MSA[i][j] != "_"]

        
        print_message = False
        message = " --- Invalid state --- "
        for i in range(len(bit_string)//self.initial_MSA.shape[1]):
            if np.sum(bit_string[i*self.initial_MSA.shape[1]:(i+1)*self.initial_MSA.shape[1]]) > 1:

                #message += f"** {i+1}'th letter placed in more than one column. \n"
                print_message = True
            if np.sum(bit_string[i*self.initial_MSA.shape[1]:(i+1)*self.initial_MSA.shape[1]]) == 0:
                #message += f"** {i+1}'th letter not placed in any column. \n"
                print_message = True
        if np.sum(bit_string) != len(letters_in_mat): 
            #message += "** Number of ones doesn't correspond to number of letters."
            print_message = True
        if print_message: print(message); print("(Returning 2D zero array)")

        new_mat = np.zeros((self.initial_MSA.shape),dtype=object)
        if print_message == False:
            counts, letters, gap = [], [], "_"
            for row in range(self.initial_MSA.shape[0]):
                current_count = 0
                current_letters  = []
                for col in range(self.initial_MSA.shape[1]):
                    if self.initial_MSA[row][col] != gap: 
                        current_count += 1
                        current_letters.append(self.initial_MSA[row][col])
                letters.append(current_letters)
                counts.append(current_count)

            lower = 0
            multiplier, regs = self.initial_MSA.shape[1], []
            for value in counts:
                for i in range(value):
                    regs.append(bit_string[lower + i * multiplier : lower + (i + 1) * multiplier])
                lower += value * multiplier

            counter = 0
            for i in range(len(letters)):
                for j in range(len(letters[i])):
                    col_idx = np.where(regs[counter] == 1)[0][0]
                    new_mat[i][col_idx] = letters[i][j]
                    counter += 1

            new_mat[new_mat == 0] = "_"
        return new_mat
    
    def index_state(self , M: int , n: int , s: int , i: int , Ns_vals: np.ndarray) -> int:
        """ Function for computing the index 'j' in bitstring, corresponding to
        given combination of (s,n,i) provived known dimensions of MSA matrix and 
        known number of letters in each row.

        (Indexing n,s,i from 1,... and adding N_0 = 0 to start of N_s vals).

        Parameters:
        -----------
            M: integer - number of columns in MSA matrix

            s: integer - 's' from (s,n,i) in x_(s,n,i)
            n: integer - 'n' from (s,n,i) in x_(s,n,i)
            i: integer - 'i' from (s,n,i) in x_(s,n,i)

            Ns_vals: np.ndarray - nr of letters in each 
                                  row of MSA, with added 0, [0,N_s1,N_s2,...]
        Returns:
        --------
            j: integer - index of bitstring

        """
        assert type(M) == int, f"Provided 'M' as wrong type: {type(M)}, should be int"
        assert type(s) == int, f"Provided 's' as wrong type: {type(s)}, should be int"
        assert type(n) == int, f"Provided 'n' as wrong type: {type(n)}, should be int"
        assert type(i) == int, f"Provided 'i' as wrong type: {type(i)}, should be int"
        assert type(Ns_vals) is np.ndarray, f"Provided 'Ns_vals' as wrong type{type(Ns_vals)}, should be {np.ndarray} "

        j = M * ((n-1) + np.sum(Ns_vals[:s])) + i - 1

        return j

    def index_weight(self , s1: int , n1: int , s2: int , n2: int , M: int):
        """ Function for computing the index 'i,j' in weightmatrix, corresponding to
        given combination of (s1,n1,s2,n2) provived known dimensions of MSA matrix and 
        known number of letters in each row.

        (Indexing n,s,i from 1,... and adding N_0 = 0 to start of N_s vals)
        Parameters:
        -----------
            M: integer - number of columns in MSA matrix

            s1: integer - 's1' from (s1,n1,s2,n2) in w_(s1,n1,s2,n2)
            n1: integer - 'n1' from (s1,n1,s2,n2) in w_(s1,n1,s2,n2)
            s2: integer - 's2' from (s1,n1,s2,n2) in w_(s1,n1,s2,n2)
            n2: integer - 'n2' from (s1,n1,s2,n2) in w_(s1,n1,s2,n2)

        Returns:
        --------
            w_mat_row_idx: integer - row index of weight matrix
            w_mat_vol_idx: integer - col index of weight matrix

        """
        assert type(M)  == int, f"Provided 'M' as wrong type: {type(M)}, should be int"
        assert type(s1) == int, f"Provided 's1' as wrong type: {type(s1)}, should be int"
        assert type(n1) == int, f"Provided 'n1' as wrong type: {type(n1)}, should be int"
        assert type(s2) == int, f"Provided 's2' as wrong type: {type(s2)}, should be int"
        assert type(n2) == int, f"Provided 'n2' as wrong type: {type(n2)}, should be int"

        w_mat_row_idx = (n1 - 1) + (s2-2) * M + (s1-1) * M
        w_mat_col_idx = n2 - 1

        return w_mat_row_idx, w_mat_col_idx
    
    def score(self , n1: int , n2: int) -> int:
        """ Function for computing the score of
            some alignment of the two chars n1, n2

        Parameters:
        -----------
            n1: any str in {"_","A","T","C","G"}
            n2: any str in {"_","A","T","C","G"}

        Returns:
        --------
            score: int in {-1,0,1}
        """
        assert type(n1) == str, f"Provided 'n1' as wrong type: {type(n1)}, should be str"
        assert type(n2) == str, f"Provided 'n2' as wrong type: {type(n2)}, should be str"

        score = None
        gap   = "_"

        if   n1 == gap or n2 == gap: score =  0; return score
        elif n1 == n2              : score = -1; return score
        elif n1 != n2              : score =  2; return score

    def encode_score_weights(self):
        """
        Encoding the score of all possible alignments
            for all n1, n2 for all s1 < s2 score(n1,n2)

        Returns:
        --------
            weight_matrices: 2D numpy array, e.g.:array([[w_1121, w_1122],
                                                         [w_1221, w_1222]])
        """
        L, C = self.initial_MSA.shape
        weight_matrices = [np.zeros((C,C)) for i in range(int(1/2 * (L - 1) * L))]
        for row1 in range(0 , self.initial_MSA.shape[0]):
            for row2 in range(row1 + 1 , self.initial_MSA.shape[0]):
                for idx1, n1 in enumerate(self.initial_MSA[row1,:]):
                    for idx2, n2 in enumerate(self.initial_MSA[row2,:]):
                        weight_matrices[row1+row2-1][idx1][idx2] = self.score(n1,n2)
        return np.vstack(np.array(weight_matrices))

    def encode_QUBO(self,normalize_weights):
        """ Function for encoding cost function into x^TQx+h^T+d form
        
        Returns:
        --------
            Q: np.ndarray - 2D numpy array 
            h: np.ndarray - 1D numpy array 
            d: float
        """

        nr_rows,nr_cols = self.initial_MSA.shape[0], self.initial_MSA.shape[1]
        nr_letters = np.array([0] + [len(np.where(self.initial_MSA[i] != "_")[0]) for i in range(nr_rows)])
        vector_dim = np.sum(nr_letters) * nr_cols

        p1,p2,p3   = self.penalties[0], self.penalties[1], self.penalties[2]

        Q = p1 * np.identity(vector_dim)
        h = -2 * p1 *np.ones(vector_dim).reshape((vector_dim,1))
        d = p1 * np.sum(nr_letters)
            
        ## Weight terms
        for s1 in range(1 , nr_rows + 1):
            for s2 in range(s1 + 1, nr_rows + 1):
                for n1 in range(1 , nr_letters[s1]+ 1):
                    for n2 in range(1 , nr_letters[s2] + 1):
                        for i in range(1 , nr_cols + 1):
                            xsni1_idx = self.index_state(nr_cols, n1, s1, i, nr_letters)
                            xsni2_idx = self.index_state(nr_cols, n2, s2, i, nr_letters)
                            row , col = self.index_weight(s1, n1, s2, n2, nr_cols)
                            Q[xsni1_idx][xsni2_idx] += self.score_weights[row][col]

        ## Remaing part of squared terms in one col pr. letter
        for s in range(1 , nr_rows + 1):
            for n in range(1 , nr_letters[s] + 1):
                for i in range(1 , nr_cols + 1):
                    for j in range(i + 1, nr_cols + 1):
                        xsni1_idx = self.index_state(nr_cols, n, s, i, nr_letters)
                        xsni2_idx = self.index_state(nr_cols, n, s, j, nr_letters)
                        Q[xsni1_idx][xsni2_idx] += 2 * p1

        ## One letter pr. ith col terms
        for s in range(1 , nr_rows + 1):
            for i in range(1 , nr_cols + 1):
                for n1 in range(1 , nr_letters[s]+ 1):
                    for n2 in range(n1 + 1 , nr_letters[s] + 1):
                        xsni1_idx = self.index_state(nr_cols, n1, s, i, nr_letters)
                        xsni2_idx = self.index_state(nr_cols, n2, s, i, nr_letters)
                        Q[xsni1_idx][xsni2_idx] += 1 * p2
        
        ## Ordering term
        for s in range(1 , nr_rows + 1):
            for n1 in range(1 , nr_letters[s]+ 1):
                for n2 in range(n1 + 1 , nr_letters[s] + 1):
                    for i1 in range(1 , nr_cols + 1):
                        for i2 in range(i1 + 1 , nr_cols + 1):
                            xsni1_idx = self.index_state(nr_cols, n1, s, i2, nr_letters)
                            xsni2_idx = self.index_state(nr_cols, n2, s, i1, nr_letters)
                            Q[xsni1_idx][xsni2_idx] += 1 * p3

        #if normalize_weights:
        #    Q *= 1./np.max([np.max(Q),np.max(h)])
        #    h *= 1./np.max([np.max(Q),np.max(h)])
            
        return Q,h,d

    def encode_Ising(self,normalize_weights):
        """ Function that transforms the cost model from
            acting as x^TQx+h^Tx+d on x in {0,1}^n to 
            s^TJs+g^Ts+c on s in {-1,1}^n

        Returns:
        --------
            J: np.ndarray - 2D numpy array 
            g: np.ndarray - 1D numpy array 
            c: float
        """

        Q, h, d = self.QUBO_model

        J = Q / 4.

        g = np.zeros((h.shape))
        for i in range(Q.shape[0]):
            g -= (Q[:,i].reshape((Q.shape[0],1)) + (Q.T)[:,i].reshape((Q.shape[0],1))) / 4.

        g -=  h / 2

        c = np.sum(Q) / 4. + np.sum(h) / 2. + d

        if normalize_weights:
            J *= 1./np.max([np.max(J),np.max(g)])
            g *= 1./np.max([np.max(J),np.max(g)])

        return J, g, c
