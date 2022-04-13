import numpy as np
import matplotlib.pyplot as plt


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


def plot_count_histogram(counts, solutions, top_number = 55):
        
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
            for idx, state in enumerate([string_to_arr(sorted_states[i]).flatten() for i in range(len(sorted_states))]):
                equal = True
                for int_idx, integer in enumerate(state.astype(np.float64)):
                    if integer != solution[int_idx]:
                        equal = False
                if equal: good_indexes.append(idx)

        ## Plotting
        fig, ax = plt.subplots(1,1,figsize=(25,15))

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
        fig.subplots_adjust(bottom=0.2) ## Increasing space below fig (in case of large states)

        plt.savefig("State histogram.png")