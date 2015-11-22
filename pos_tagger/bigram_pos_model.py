class BigramPosModel:
    def __init__(self, corpus):
        self.corpus = corpus

    def calculate_bigram_pos_frequencies(self):
        pos = {}
        for i, tagged_word in enumerate(self.corpus.tagged_words[:-1]):
            _, this_pos = tagged_word
            _, next_pos = self.corpus.tagged_words[i+1]
            if this_pos not in pos.keys():
                pos[this_pos] = {}
            if next_pos not in pos[this_pos].keys():
                pos[this_pos][next_pos] = 0
            pos[this_pos][next_pos] += 1
        return pos

    def calculate_word_pos_frequencies(self):
        words = {}
        for word, pos in self.corpus.tagged_words:
            if word not in words.keys():
                words[word] = {}
            if pos not in words[word].keys():
                words[word][pos] = 0
            words[word][pos] += 1
        return words
