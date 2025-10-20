"""
Posterior Decoding for Hidden Markov Models

This module implements the forward-backward algorithm (posterior decoding)
for finding the most likely state at each position given an observation
sequence and an HMM.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray


def posterior_decode(observation_seq: List[int], hmm) -> List[str]:
    """
    Decode an observation sequence using posterior (forward-backward) decoding.

    INPUT:
    - observation_seq: List of observations (integers indexing alphabet)
    - hmm: HMM object with states, initial_state_probs, transition_matrix,
      and emission_matrix

    OUTPUT:
    - state_seq: List containing the sequence of most likely states (state names)

    IMPLEMENTATION NOTES:
    1) The forward and backward matrices are implemented as matrices that are
       transposed relative to the way they are shown in class. Rows correspond
       to observations and columns to states.
    2) After computing the forward/backward probabilities for each observation,
       they are normalized by dividing by their sum. This maintains
       proportionality while avoiding numerical underflow.
    3) The posterior probability matrix is the element-wise product of the
       forward and backward matrices, normalized at each observation.
    """
    # YOUR CODE HERE
    # pass the forward and backward matrices into _poster_probabilities
    posterior_prob_matrix = _posterior_probabilities(observation_seq, hmm)

    # initialize variables
    number_of_observations = len(observation_seq)
    decoded_states = []

    # decode the posterior prob matrix by assigning the state that corresponds to the max value
    for obs_i in range(number_of_observations):
        # find index of max prob
        max_index = np.argmax(posterior_prob_matrix[obs_i, :])
        # map index to state
        decoded_states.append(hmm.states[max_index])
    return decoded_states

# DEFINE _posterior_probabilites here
def _posterior_probabilities(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the posterior probability matrix
    """
    # Build the a and B matrices 
    forward_matrix = _build_forward_matrix(observation_seq, hmm)
    backward_matrix = _build_backward_matrix(observation_seq, hmm)

    # Initialize variables
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states
    posterior_prob_matrix =  np.zeros((number_of_observations, number_of_states))

    # Calculate the posterior probability for each observation and state
    for obs_i in range(number_of_observations):
        for state_j in range(number_of_states):
            # posterior prob = a*B / sum(all a and B for obs) (this is normalized)
            num = forward_matrix[obs_i, state_j] * backward_matrix[obs_i, state_j]
            denom = sum(forward_matrix[obs_i, :] * backward_matrix[obs_i, :]) 
            posterior_prob_matrix[obs_i, state_j] = num/denom
            
    return posterior_prob_matrix

def _build_forward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the forward probability matrix.

    Similar to Viterbi but uses sum instead of max.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize alpha matrix
    forward_matrix = np.zeros((number_of_observations, number_of_states))

    # loop over all observations
    for obs_i in range(number_of_observations):
        # the current observation
        obs_letter = observation_seq[obs_i]  

        # loop over all states
        for state_j in range(number_of_states):
            # calculate initial state probs
            if obs_i == 0:
                forward_matrix[obs_i, state_j] = (
                    hmm.initial_state_probs[state_j] * hmm.emission_matrix[obs_letter][state_j]
                )
            # otherwise calculate the state prob as the sum of the incoming nodes
            else:
                # all incoming node probabilities
                prev_nodes = []
                for prev_state in range(number_of_states):
                    prev_nodes.append(
                        forward_matrix[obs_i - 1, prev_state]
                        * hmm.transition_matrix[prev_state][state_j]
                    )

                # the alpha value is the sum of all incoming nodes times emission prob
                forward_matrix[obs_i, state_j] = np.sum(prev_nodes) * hmm.emission_matrix[obs_letter][state_j]

        # Normalize to avoid underflow
        total = np.sum(forward_matrix[obs_i, :])
        if total > 0:
            forward_matrix[obs_i, :] = forward_matrix[obs_i, :] / total
        else:
            # Handle impossible observation
            forward_matrix[obs_i, :] = np.nan

    return forward_matrix


def _build_backward_matrix(observation_seq: List[int], hmm) -> NDArray[np.float64]:
    """
    Build the backward probability matrix.

    Works backwards from the last observation.
    Returns a matrix where rows are observations and columns are states.
    Each entry is normalized to avoid underflow.
    """
    # YOUR CODE HERE
    number_of_observations = len(observation_seq)
    number_of_states = hmm.num_states

    # Initialize backward matrix
    backward_matrix = np.zeros((number_of_observations, number_of_states))

    # Initializate the last observation to be = 1
    backward_matrix[-1, :] = 1.0
    # Normalize the last row
    backward_matrix[-1, :] = backward_matrix[-1, :] / np.sum(backward_matrix[-1, :])

    # loop backwards through all previous observations
    for obs_i in range(number_of_observations - 2, -1, -1):
        # the next observation (t + 1)
        next_obs = observation_seq[obs_i + 1]

        # loop over all state transitions for the current obs
        for state_i in range(number_of_states):
            next_nodes = []
            for next_state in range(number_of_states):
                next_nodes.append(
                    hmm.transition_matrix[state_i][next_state] # transitions go forwards (current > next)
                    * hmm.emission_matrix[next_obs][next_state] # IMP: use emission from next state
                    * backward_matrix[obs_i + 1, next_state] # get the previous B
                )

            # the beta value is the sum of all nodes
            backward_matrix[obs_i, state_i] = sum(next_nodes)

        # Normalize to avoid underflow
        total = np.sum(backward_matrix[obs_i, :])
        if total > 0:
            backward_matrix[obs_i, :] = backward_matrix[obs_i, :] / total
        else:
            # Handle impossible observation
            backward_matrix[obs_i, :] = np.nan

    return backward_matrix


def _max_position(list_of_numbers: NDArray[np.float64]) -> int:
    """
    Find the index of the maximum value in a list.

    Returns the first index if there are ties or extremely close values.
    """
    max_value = -np.inf
    max_position = 0

    for i, value in enumerate(list_of_numbers):
        # This handles extremely close values that arise from numerical instability
        if value / max_value > 1 + 1E-5 if max_value > 0 else value > max_value:
            max_value = value
            max_position = i

    return max_position
