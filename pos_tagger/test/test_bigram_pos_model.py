from pos_tagger.bigram_pos_model import BigramPosModel
from pos_tagger.corpus import TrainingBrownCorpus, TuningBrownCorpus


class TestBigramPosModel:

    model = BigramPosModel(TrainingBrownCorpus())
    tuning_data = TuningBrownCorpus()

    def test_bigram_pos_model(self):
        print("length of tuning data:", len(self.tuning_data.tagged_sentences))
        mistakes = 0
        for i, sentence in enumerate(self.tuning_data.tagged_sentences):
            untagged_sentence = [word_pos[0] for word_pos in sentence]
            tags = [word_pos[1] for word_pos in sentence]
            result = self.model.decode(untagged_sentence)
            if tags != result:
                mistakes += 1
                print(i, mistakes, " ".join(untagged_sentence))
                print(" ".join(tags))
                print(" ".join(result))
        print("length of tuning data:", len(self.tuning_data.tagged_sentences))
        print("number of mistakes:", mistakes)
