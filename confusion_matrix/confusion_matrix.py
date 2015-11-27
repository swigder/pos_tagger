class ConfusionMatrix:
    def __init__(self, function, possible_results):
        self.function = function
        self.possible_results = possible_results

    def build(self, test_data):
        confusions = dict()
        for input, correct_value in test_data:
            model_value = self.function(input)
            if model_value is list:
                for correct, model in zip(correct_value, model_value):
                    self.evaluate_and_record(confusions, correct, model)
            else:  # scalar
                self.evaluate_and_record(confusions, correct_value, model_value)
        return confusions

    def evaluate_and_record(self, confusions, correct, model):
        if correct == model:
            return
        if correct not in confusions:
            confusions[correct] = dict()
        if model not in confusions[correct]:
            confusions[correct][model] = 0
        confusions[correct][model] += 1

    def format(self, confusions):
        table = ' ' * 5
        for possible_result in self.possible_results:  # table header
            table += '{0:5}'.format(possible_result)
        for possible_result_row in self.possible_results:
            table += '\n'
            table += '{0:5}'.format(possible_result_row)
            if possible_result_row in confusions:
                for possible_result_value in self.possible_results:
                    if possible_result_value in confusions[possible_result_row]:
                        table += '{0:<5}'.format(confusions[possible_result_row][possible_result_value])
                    else:
                        table += '{0:<5}'.format(0)
            else:
                table += '{0:<5}'.format(0) * len(self.possible_results)
        return table
