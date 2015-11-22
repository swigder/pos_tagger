from pos_tagger.corpus import Corpus


class TestCorpus:

    sentences = [
        [('I', 'PRON'), ('need', 'VERB'), ('to', 'PRT'), ('test', 'VERB'), ('.', '.')],
        [('Sometimes', 'ADV'), ('I', 'PRON'), ("don't", 'VERB'), ('want', 'VERB'), ('to', 'PRT'), ('.', '.')],
        [('But', 'CONJ'), ('I', 'PRON'), ('MUST', 'VERB'), ('!', '.')],
    ]

    def test_corpus_lowercases_words(self):
        corpus = Corpus(self.sentences)
        assert corpus.tagged_sentences == [
            [('i', 'PRON'), ('need', 'VERB'), ('to', 'PRT'), ('test', 'VERB'), ('.', '.')],
            [('sometimes', 'ADV'), ('i', 'PRON'), ("don't", 'VERB'), ('want', 'VERB'), ('to', 'PRT'), ('.', '.')],
            [('but', 'CONJ'), ('i', 'PRON'), ('must', 'VERB'), ('!', '.')],
        ]
