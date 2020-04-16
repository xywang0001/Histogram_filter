import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """
        
    def histogram_filter(self, cmap, belief, action, observation):
        """
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
        """
        ### Your Algorithm goes Below.
        
        #cmap = np.rot90(cmap)
        #belief = np.rot90(belief)
        n = cmap.shape[0]
        m = cmap.shape[1]
        ob_true = 0.9
        ob_false = 0.1
        p_move = 0.9
        p_stay = 0.1        
        
        # update by action
        M = np.zeros([n,m])
        M[19][19] = 1
        for i in range((n-1)):
            M[i][i] = p_stay
            M[i][i+1] = p_move

        M2 = np.zeros([n,m])
        M2[0][0] = 1
        for i in range((n-1)):
            M2[19-i][19-i] = p_stay
            M2[19-i][19-i-1] = p_move
            
        if action[0] == 1:
            n_belief = np.dot(belief, M)
        elif action[0] == -1:
            n_belief = np.dot(belief, M2)      
        elif action[1] == -1:
            n_belief = np.transpose(np.dot(np.transpose(belief),M))
        elif action[1] == 1:
            n_belief = np.transpose(np.dot(np.transpose(belief), M2))
               
     
        # update by observation
       
        s_prob = np.zeros([n,m])
        for i in range(n):
            for j in range(m):
                bool_ = int(cmap[i][j]==observation)
                s_prob[i][j] = bool_*ob_true + (1-bool_)*ob_false
        n_belief = np.multiply(s_prob, n_belief)
        
        n_belief = np.array(n_belief)
        #n_belief = np.rot90(n_belief)
        pos = np.unravel_index(np.argmax(n_belief),n_belief.shape)
        pos = np.array([pos[1],n-1-pos[0]])
        
        tuple_obj = (n_belief, pos)

        
        return tuple_obj
        
        
        
        
        
        