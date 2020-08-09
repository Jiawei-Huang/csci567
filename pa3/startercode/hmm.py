from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        Oindex = [self.obs_dict[i] for i in Osequence]
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s, Oindex[0]] 
        
        for t in range(1, L):
            for s in range(S):
                alpha[s, t] = self.A[: , s].dot(alpha[:, t-1]) * self.B[s, Oindex[t]]
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        Oindex = [self.obs_dict[i] for i in Osequence]
        beta[:, L-1] = 1
        for t in range(L-2, -1, -1):
            for s in range(S):
                beta[s, t] = (self.A[s, :] * self.B[:, Oindex[t+1]]).dot(beta[:, t+1])
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        sequence_prob = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        prob = alpha*beta/sequence_prob
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        Oindex = [self.obs_dict[i] for i in Osequence]
        sequence_prob = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        temp = self.B[:, Oindex[1:]]*beta[:, 1:]

        for s in range(S):
            alpha_vec = alpha[s,:L-1].reshape((-1, L-1))
            A_vec = self.A[s,:].reshape((S, -1))
            prob[s,:,:] = alpha_vec * temp
            prob[s,:,:] = A_vec * prob[s,:,:]
        prob = prob / sequence_prob
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        Oindex = []
        for i in Osequence:
            if i not in self.obs_dict.keys():
                self.obs_dict[i] = len(self.obs_dict.keys())
                self.B = np.concatenate((self.B, np.zeros((self.B.shape[0], 1)) + 1e-6), axis=1)
            Oindex.append(self.obs_dict[i])

        delta = self.pi * self.B[:, Oindex[0]]
        track = np.zeros((S, L-1))

        for t in range(1, L):
            temp = self.A * delta.reshape((S, -1))
            track[:, t-1] = np.argmax(temp, axis=0)
            delta = self.B[:, Oindex[t]] * np.max(temp, axis=0)
        
        track = track.astype(int)
        index = np.argmax(delta)
        path.append(index)
        for i in range(L-2, -1, -1):
            index = track[index, i]
            path.append(index)
        path.reverse()

        path = [list(self.state_dict.keys())[i] for i in path]
        ###################################################
        return path
