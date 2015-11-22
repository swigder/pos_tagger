from nltk.corpus import brown


class Corpus:
    """
    Corpus wrapper that normalizes a pos-tagged corpus by setting all words to lowercase
    """

    def __init__(self, tagged_words):
        """
        :param tagged_words: list of tuples, where each tuple is of form (word, pos_tag)
        """
        self.tagged_words = [(word_pos[0].lower(), word_pos) for word_pos in tagged_words]


class BrownCorpus(Corpus):
    """
    Wrapper over the Brown Corpus using universal tagset that keeps all data in memory for performance improvements
    """

    def __init__(self):
        super().__init__(brown.tagged_words(tagset='universal'))


class TrainingBrownCorpus(BrownCorpus):
    """
    Wrapper over a subset of the Brown Corpus to be used for training (use in conjunction with TestBrownCorpus in this
    file)
    """

    def __init__(self):
        super().__init__()
        self.tagged_words = self.tagged_words[:int(len(self.tagged_words) * .9)]


class TestBrownCorpus(BrownCorpus):
    """
    Wrapper over a subset of the Brown Corpus to be used for testing (use in conjunction with TrainingBrownCorpus in
    this file)
    """

    def __init__(self):
        super().__init__()
        self.tagged_words = self.tagged_words[int(len(self.tagged_words) * .9):]
