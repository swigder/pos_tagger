from utilities.utilities import binary_search


class HiddenMarkovModel:
    """
    Implementation of a Hidden Markov Model given counts of hidden states and observations as seen in training data
    """

    START = "<START>"
    
    def __init__(self, hidden_bigram_frequencies, observation_frequencies, state_counts, observations):
        """
        :param hidden_bigram_frequencies: map of maps, where outside map is prior state, inside map is current state,
        and value is bigram count of prior state, current state
        :param observation_frequencies: map of maps, where outside map is state, inside map is observations, and value
        is count of the given observation in the given state
        :param state_counts: map of states to the number of times that state occurs (for converting counts to
        probabilities)
        """
        self.hidden_bigram_frequencies = hidden_bigram_frequencies
        self.observation_frequencies = observation_frequencies
        self.state_counts = state_counts
        self.states = [state for state in state_counts.keys() if state != self.START]
        self.observations = sorted(observations)

    def decode(self, observations):
        """
        Decodes a given observation into the most likely sequence of hidden states, using Viterbi
        :param observations: list of observations
        :return: most probable list of hidden states
        """
        matrix = [{}]
        for state in self.states:
            probability_state_given_previous = self.get_bigram_probability(self.START, state)
            probability_word_given_state = self.get_observed_probability(state, observations[0])
            probability = probability_state_given_previous * probability_word_given_state
            backpointer = None
            if probability > 0:
                matrix[0][state] = (probability, backpointer)
        for i, observation in enumerate(observations[1:], 1):
            matrix.append({})
            for state in self.states:
                probability_observation_given_state = self.get_observed_probability(state, observation)
                if probability_observation_given_state == 0:
                    continue
                best_probability, best_backpointer = 0, None
                for previous_state, previous_value in matrix[i-1].items():
                    probability_previous, _ = previous_value
                    probability_current_given_previous = self.get_bigram_probability(previous_state, state)
                    current_probability = probability_previous * probability_current_given_previous
                    if current_probability > best_probability:
                        best_probability, best_backpointer = current_probability, previous_state
                probability = best_probability * probability_observation_given_state
                if probability > 0:
                    matrix[i][state] = (probability, best_backpointer)

        # reconstruct the best path
        best_probability = 0
        backpointer = None
        for state, value in matrix[-1].items():
            probability, _ = value
            if probability > best_probability:
                best_probability = probability
                backpointer = state
        if best_probability == 0:
            print("No possible result found for observations", observations)
            return ['<UNK>'] * len(observations)
        states = []
        for i in range(len(observations), 0, -1):
            states.append(backpointer)
            _, backpointer = matrix[i-1][backpointer]

        states.reverse()
        return states

    def get_bigram_probability(self, prior, state):
        """
        Gets the probability of a state given a prior state
        :param state: the current state
        :param prior: the previous state
        :return: probability of current state given prior state, defined as as count(prior, current) / count(prior)
        """
        priors = self.hidden_bigram_frequencies[prior]
        return priors[state] / self.state_counts[prior] if state in priors else 0

    def get_observed_probability(self, state, observation):
        """
        Gets the probability of an observation given a state.  This is not a true probability, as observations that are
        not part of the list of observations will be given a weight of one, to facilitate the HMM considering only
        hidden states in case of an unknown observation
        :param state: the current state of the model
        :param observation: the observation
        :return: probability of the observation given the state, defined as count(observation, state) / count(state), or
        1 if the observation does not exist in the list of possible observations
        """
        if binary_search(self.observations, observation) == -1:
            return 1
        observations = self.observation_frequencies[state]
        return observations[observation] / self.state_counts[state] if observation in observations else 0
