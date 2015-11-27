from confusion_matrix.confusion_matrix import ConfusionMatrix


class TestConfusionMatrix:

    possible_values = ['a', 'b', 'c']
    test_data = [(1, 'a'), (2, 'b'), (3, 'c')]
    confusion_matrix = ConfusionMatrix(lambda x: {1: 'a', 2: 'c', 3: 'b'}[x], possible_values)

    def test_confusion_matrix(self):
        assert {'b': {'c': 1}, 'c': {'b': 1}} == self.confusion_matrix.build(self.test_data)

    def test_format(self):
        assert "     a    b    c    \n" \
               "a    0    0    0    \n" \
               "b    0    0    1    \n" \
               "c    0    1    0    " \
               == self.confusion_matrix.format({'b': {'c': 1}, 'c': {'b': 1}})
