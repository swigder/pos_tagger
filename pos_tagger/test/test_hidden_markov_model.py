from pos_tagger.hidden_markov_model import HiddenMarkovModel


class TestHiddenMarkovModel:
    hidden_frequencies = {
        '<START>': {'HOT': 8, 'COLD': 2},
        'HOT': {'HOT': 7, 'COLD': 3},
        'COLD': {'HOT': 4, 'COLD': 6}
    }
    observed_frequencies = {
        'HOT': {1: 2, 2: 4, 3: 4},
        'COLD': {1: 5, 2: 4, 3: 1}
    }
    states = {
        '<START>': 10,
        'HOT': 10,
        'COLD': 10
    }
    model = HiddenMarkovModel(hidden_frequencies, observed_frequencies, states)

    def test_hidden_markov_model(self):
        assert self.model.decode([3, 1, 3]) == ['HOT', 'HOT', 'HOT']
