from confusion_matrix.confusion_matrix import ConfusionMatrix
from pos_tagger.bigram_pos_model import BigramPosModel
from pos_tagger.corpus import BrownCorpus, TrainingBrownCorpus, TuningBrownCorpus, TestBrownCorpus
from pos_tagger.most_frequent_tag_pos_model import MostFrequentTagPosModel


def q1_train_using_brown_corpus():
    corpus = BrownCorpus()
    model = BigramPosModel(corpus)


def q2_construct_training_tuning_test_set():
    training = TrainingBrownCorpus()
    tuning = TuningBrownCorpus()
    test = TestBrownCorpus()


def q3_train_on_tagged_corpus():
    # unclear how this differs from 1?
    corpus = TrainingBrownCorpus()
    model = BigramPosModel(corpus)


def q4_write_program_pos_tag():
    print("Please run the program from command line:  python3 postag.py SENTENCE")


def q5_error_rate_on_test_set():
    training = TrainingBrownCorpus()
    model = BigramPosModel(training)
    test = TestBrownCorpus()
    # todo actually do something


def q6_error_rate_most_frequent_tag():
    training = TrainingBrownCorpus()
    model = MostFrequentTagPosModel(training)
    test = TestBrownCorpus()
    # todo actually do something


def q7_confusion_matrix():
    training = TrainingBrownCorpus()
    model = BigramPosModel(training)
    test = TestBrownCorpus()
    confusion_matrix = ConfusionMatrix(lambda x: model.decode(x), model.states)
    matrix = confusion_matrix.build([zip(*sentence) for sentence in test.tagged_sentences])
    print(confusion_matrix.format(matrix))

if __name__ == '__main__':
    q7_confusion_matrix()
