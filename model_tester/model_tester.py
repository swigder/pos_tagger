class ModelTester:
    def __init__(self, function, possible_results, test_data):
        self.function = function
        self.possible_results = possible_results
        self.test_data = test_data
        self._build()

    def _build(self):
        self.results = dict()
        for input, correct_value in self.test_data:
            model_value = self.function(input)
            if type(model_value) is list:
                for correct, model in zip(correct_value, model_value):
                    self._record(correct, model)
            else:  # scalar
                self._record(correct_value, model_value)

    def _record(self, correct, model):
        if correct not in self.results:
            self.results[correct] = dict()
        if model not in self.results[correct]:
            self.results[correct][model] = 0
        self.results[correct][model] += 1

    def format_confusion_table(self):
        table = ' ' * 5
        for possible_result in self.possible_results:  # table header
            table += '{0:5}'.format(possible_result)
        for possible_result_row in self.possible_results:
            table += '\n'
            table += '{0:5}'.format(possible_result_row)
            if possible_result_row in self.results:
                for possible_result_value in self.possible_results:
                    if possible_result_value in self.results[possible_result_row]:
                        table += '{0:<5}'.format(self.results[possible_result_row][possible_result_value])
                    else:
                        table += '{0:<5}'.format(0)
            else:
                table += '{0:<5}'.format(0) * len(self.possible_results)
        return table

    def get_error_rate(self):
        total = 0
        error = 0
        for correct_value, model_values in self.results.items():
            for model_value, instances in model_values.items():
                total += instances
                if model_value != correct_value:
                    error += instances
        return error / total