from nltk.corpus import brown


class Corpus:
    """
    Corpus wrapper that normalizes a pos-tagged corpus by setting all words to lowercase
    """

    def __init__(self, tagged_sentences):
        """
        :param tagged_sentences: list of lists of tuples, where each list is a sentence and each tuple is a token in the
        sentence of form (word, pos_tag)
        """
        self.tagged_sentences = [[(word_pos[0].lower(), word_pos[1]) for word_pos in sentence]
                                 for sentence in tagged_sentences]
        self.words = [word_pos[0] for sentence in tagged_sentences for word_pos in sentence]


class BrownCorpus(Corpus):
    """
    Wrapper over the Brown Corpus using universal tagset that keeps all data in memory for performance improvements
    """

    def __init__(self):
        super().__init__(brown.tagged_sents(tagset='universal'))


class TrainingBrownCorpus(BrownCorpus):
    """
    Wrapper over a subset of the Brown Corpus to be used for training (use in conjunction with TestBrownCorpus in this
    file)
    """

    def __init__(self):
        super().__init__()
        self.tagged_sentences = self.tagged_sentences[:int(len(self.tagged_sentences) * .8)]


class TuningBrownCorpus(BrownCorpus):
    """
    Wrapper over a subset of the Brown Corpus to be used for testing (use in conjunction with TrainingBrownCorpus in
    this file)
    """

    def __init__(self):
        super().__init__()
        sentences = len(self.tagged_sentences)
        self.tagged_sentences = self.tagged_sentences[int(sentences * .8):int(sentences * .9)]


class TestBrownCorpus(BrownCorpus):
    """
    Wrapper over a subset of the Brown Corpus to be used for testing (use in conjunction with TrainingBrownCorpus in
    this file)
    """

    def __init__(self):
        super().__init__()
        self.tagged_sentences = self.tagged_sentences[int(len(self.tagged_sentences) * .9):]
