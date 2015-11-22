class HiddenMarkovModel:

    START = "<START>"
    
    def __init__(self, hidden_bigram_frequencies, observation_frequencies, state_counts):
        self.hidden_bigram_frequencies = hidden_bigram_frequencies
        self.observation_frequencies = observation_frequencies
        self.state_counts = state_counts
        self.states = [state for state in state_counts.keys() if state != self.START]

    def decode(self, observations):
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
        states = None
        for state in self.states:
            probability, _ = matrix[-1][state]
            if probability > best_probability:
                best_probability = probability
                states = [state]
        backpointer = states[0]
        for i in range(len(observations), 0, -1):
            _, backpointer = matrix[i-1][backpointer]
            states.append(backpointer)

        states.reverse()
        return states

    def get_bigram_probability(self, state, prior):
        priors = self.hidden_bigram_frequencies[prior]
        return priors[state] / self.state_counts[prior] if state in priors else 0

    def get_observed_probability(self, state, observation):
        observations = self.observation_frequencies[state]
        return observations[observation] / self.state_counts[state] if observation in observations else 0
