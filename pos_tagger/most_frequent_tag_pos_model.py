class MostFrequentTagPosModel:
    """
    Part-of-speech tagging model that trains on a tagged corpus and whose decoding method returns each word's most
    frequent tag in the training data.
    """
    def __init__(self, corpus):
        """
        :param corpus: tagged corpus of sentences of form list of sentences, where each sentence is a list of tuples of
        form (word, pos)
        """
        self.corpus = corpus
        self.most_common_pos = self._calculate_most_common_pos_per_word()

    def _calculate_most_common_pos_per_word(self):
        """
        Calculates the most common tag for each word in the corpus
        :return: dict of words to their most frequent tag
        """
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
        """
        Decodes a list of words into a list of parts of speech
        :param observations: list of words
        :return: the most common tag in the training data for each word in the input
        """
        return map(lambda x: self.most_common_pos[x] if x in self.most_common_pos else '<UNK>', observations)