class HiddenMarkovModel:
    """
    Implementation of a Hidden Markov Model given counts of hidden states and observations as seen in training data
    """

    START = "<START>"
    
    def __init__(self, hidden_bigram_frequencies, observation_frequencies, state_counts):
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

    def decode(self, observations):
        """
        Decodes a given observation into the most likely sequence of hidden states, using Viterbi
        :param observations: list of observations
        :return: most probable list of hidden states
        """
        matrix = [{}]
        for state in self.states:
            probability_state_given_previous = self.get_bigram_probability(state, self.START)
            probability_word_given_state = self.get_observed_probability(state, observations[0])
            probability = probability_state_given_previous * probability_word_given_state
            backpointer = None
            matrix[0][state] = (probability, backpointer)
        for i, observation in enumerate(observations[1:], 1):
            matrix.append({})
            for state in self.states:
                probability_observation_given_state = self.get_observed_probability(state, observation)
                best_probability = 0
                best_backpointer = None
                for previous_state in self.states:
                    probability_previous, _ = matrix[i-1][previous_state]
                    probability_current_given_previous = self.get_bigram_probability(previous_state, state)
                    current_probability = probability_previous * probability_current_given_previous
                    if current_probability > best_probability:
                        best_probability = current_probability
                        best_backpointer = previous_state
                probability = best_probability * probability_observation_given_state
                matrix[i][state] = (probability, best_backpointer)
        best_probability = 0
        backpointer = None
        for state in self.states:
            probability, _ = matrix[-1][state]
            if probability > best_probability:
                best_probability = probability
                backpointer = state
        states = []
        for i in range(len(observations), 0, -1):
            states.append(backpointer)
            _, backpointer = matrix[i-1][backpointer]

        states.reverse()
        return states

    def get_bigram_probability(self, state, prior):
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
        Gets the probability of an observation given a state
        :param state: the current state of the model
        :param observation: the observation
        :return: probability of the observation given the state, defined as count(observation, state) / count(state)
        """
        observations = self.observation_frequencies[state]
        return observations[observation] / self.state_counts[state] if observation in observations else 0
