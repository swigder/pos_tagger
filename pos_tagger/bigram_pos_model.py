from pos_tagger.hidden_markov_model import HiddenMarkovModel


class BigramPosModel(HiddenMarkovModel):

    def __init__(self, corpus):
        self.corpus = corpus
        super().__init__(self.calculate_bigram_pos_frequencies(),
                         self.calculate_pos_word_frequencies(),
                         self.calculate_state_frequencies())

    def calculate_bigram_pos_frequencies(self):
        pos = dict()
        pos[self.START] = {}
        for sentence in self.corpus.tagged_sentences:
            _, first_pos = sentence[0]
            if first_pos not in pos[self.START]:
                pos[self.START][first_pos] = 0
            pos[self.START][first_pos] += 1
            for i, tagged_word in enumerate(sentence[:-1]):
                _, this_pos = tagged_word
                _, next_pos = sentence[i+1]
                if this_pos not in pos.keys():
                    pos[this_pos] = {}
                if next_pos not in pos[this_pos].keys():
                    pos[this_pos][next_pos] = 0
                pos[this_pos][next_pos] += 1
        return pos

    def calculate_pos_word_frequencies(self):
        pos_words = dict()
        for sentence in self.corpus.tagged_sentences:
            for word, pos in sentence:
                if pos not in pos_words.keys():
                    pos_words[pos] = {}
                if word not in pos_words[pos].keys():
                    pos_words[pos][word] = 0
                pos_words[pos][word] += 1
        return pos_words

    def calculate_state_frequencies(self):
        states = dict()
        states[self.START] = 0
        for sentence in self.corpus.tagged_sentences:
            states[self.START] += 1
            for _, pos in sentence:
                if pos not in states:
                    states[pos] = 0
                states[pos] += 1
        return states
