from model_tester.model_tester import ModelTester


class TestModelTester:

    possible_values = ['a', 'b', 'c']
    test_data = [(1, 'a'), (2, 'b'), (3, 'c')]
    model_tester = ModelTester(lambda x: {1: 'a', 2: 'c', 3: 'b'}[x], possible_values, test_data)

    def test_confusion_matrix(self):
        assert {'b': {'c': 1}, 'c': {'b': 1}} == self.model_tester.results

    def test_format(self):
        assert "     a    b    c    \n" \
               "a    0    0    0    \n" \
               "b    0    0    1    \n" \
               "c    0    1    0    " \
               == self.model_tester.format_confusion_table()
