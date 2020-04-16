import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']


    #### Test your code here
    
    belief = [[1/400 for i in range(20)] for j in range(20)]
    
    H_filter = HistogramFilter()
    
    for i in range(30):
    
        p = H_filter.histogram_filter(cmap, belief, actions[i], observations[i])
        belief = p[0]
        print (p[1])
