class ConstantUnseenObservationHandler:
    """
    This unseen observation handler returns a constant non-zero probability for all observations and states, so that a
    HMM will use only the hidden probabilities, and not the observed probabilities, for unseen observations.
    """
    def get_probability(self, state, observation):
        """
        Gets the probability that a given unseen observation is of a given state
        :param state: state under consideration
        :param observation: unseen observation
        :return: nonzero constant
        """
        return 1


class ClosedClassUnseenObservationHandler:
    """
    This closed class unseen observation handler will return a 0 probability for all observations given a state in
    configured closed class states, and a constant non-zero probability for all observations for all other states.
    """
    def __init__(self, closed_states):
        """
        :param closed_states: list of closed states, where all observations of this state are expected to be seen in the
        training data
        """
        self.closed_states = closed_states

    def get_probability(self, state, observation):
        """
        Gets the probability that a given unseen observation is of a given state
        :param state: state under consideration
        :param observation: unseen observation
        :return: 0 if state is in closed_states, nonzero constant otherwise
        """
        return 0 if state in self.closed_states else 1
