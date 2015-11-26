class MostFrequentTagPosModel:
    def __init__(self, corpus):
        self.corpus = corpus
        self.most_common_pos = self.calculate_most_common_pos_per_word()

    def calculate_most_common_pos_per_word(self):
        word_pos = dict()
        for sentence in self.corpus.tagged_sentences:
            for word, pos in sentence:
                if word not in word_pos.keys():
                    word_pos[word] = {}
                if pos not in word_pos[word].keys():
                    word_pos[word][pos] = 0
                word_pos[word][pos] += 1
        most_common_pos = dict()
        for word, value in word_pos.items():
            most_common_pos[word] = max(value, key=value.get)
        return most_common_pos

    def decode(self, observations):
        return map(lambda x: self.most_common_pos[x] if x in self.most_common_pos else '<UNK>', observations)