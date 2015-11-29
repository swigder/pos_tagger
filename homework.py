from model_tester.model_tester import ModelTester
from pos_tagger.bigram_pos_model import BigramPosModel
from pos_tagger.corpus import BrownCorpus, TrainingBrownCorpus, TuningBrownCorpus, TestBrownCorpus
from pos_tagger.most_frequent_tag_pos_model import MostFrequentTagPosModel


class Homework:
    """
    Simple class that explicitly runs code to generate responses for homework questions.
    """

    def q1_train_using_brown_corpus(self):
        """
        Train using the Brown corpus as found in NLTK.
        The universal tagset was chosen, mainly because it was easier to manually test with easily understood tag names.
        """
        corpus = BrownCorpus()
        model = BigramPosModel(corpus)
        print("Q1:  Trained model on full Brown Corpus:", model)

    def q2_construct_training_tuning_test_set(self):
        """
        Construct a training and testing set from the BrownCorpus.
        A tuning set was also created and used for manual experimentation.
        """
        self.training = TrainingBrownCorpus()
        self.tuning = TuningBrownCorpus()
        self.test = TestBrownCorpus()
        print("Q2:  Constructed training, tuning, test sets:", self.training, self.tuning, self.test)

    def q3_train_on_tagged_corpus(self):
        """
        Train the transition and observation probabilities of the HMM tagger directly on tagged data.
        """
        self.model = BigramPosModel(self.training)
        print("Q3:  Trained model on training data:", self.model)

    def q4_write_program_pos_tag(self):
        """
        Write a program postag that reads a space delimited sentence from the command line and outputs a sequence of
        tags, separated by spaces.
        """
        print("Q4:  Please run the program from command line:  python3 postag.py SENTENCE")

    def q5_error_rate_on_test_set(self):
        """
        Run your algorithm on the test set and report its error rate.
        """
        self.test_data = [zip(*sentence) for sentence in self.test.tagged_sentences]
        self.model_tester = ModelTester(lambda x: self.model.decode(x), self.model.states, self.test_data)
        print("Q5:  Error rate on model for test set:", self.model_tester.get_error_rate())

    def q6_error_rate_most_frequent_tag(self):
        """
        Compare this error rate to the 'most frequent tag' baseline.
        """
        model = MostFrequentTagPosModel(self.training)
        model_tester = ModelTester(lambda x: model.decode(x), self.model.states, self.test_data)
        print("Q6:  Error rate on most-frequent-tag model for test set:", model_tester.get_error_rate())

    def q7_confusion_matrix(self):
        """
        Build a confusion matrix and investigate the most frequent errors produced by your tagger.
        """
        print("Q7:  Confusion matrix over model for test set:\n", self.model_tester.format_confusion_table())

    def do_homework(self):
        """
        Convenience method to run all homework questions.
        Note:  In order to clearly denote work done for each question, instance variables are set outside of the
        constructor, which means that later methods may be dependent on earlier methods.  Therefore, it is important to
        run all questions in order.
        """
        self.q1_train_using_brown_corpus()
        self.q2_construct_training_tuning_test_set()
        self.q3_train_on_tagged_corpus()
        self.q4_write_program_pos_tag()
        self.q5_error_rate_on_test_set()
        self.q6_error_rate_most_frequent_tag()
        self.q7_confusion_matrix()

if __name__ == '__main__':
    Homework().do_homework()
