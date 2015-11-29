class ModelTester:
    """
    Tester to evaluate models using test data and report on error rates, generate confusion matrix, etc.
    """

    def __init__(self, function, possible_results, test_data):
        """
        :param function: function to run on test data (e.g., lambda that calls model)
        :param possible_results: list of possible results (for confusion matrix)
        :param test_data: test data of form tuple (model_input, correct_output).  Input can be any format; if output is
        a list, each element of output will be evaluated separately for calculating error rates and confusion matrix.
        """
        self.function = function
        self.possible_results = possible_results
        self.test_data = test_data
        self._build()

    def _build(self):
        """
        Runs the given function on all test data and records results
        """
        self.results = dict()
        for input, correct_value in self.test_data:
            model_value = self.function(input)
            if type(model_value) is list:
                for correct, model in zip(correct_value, model_value):
                    self._record(correct, model)
            else:  # scalar
                self._record(correct_value, model_value)

    def _record(self, correct, model):
        """
        Records results for a given run of a model
        :param correct: correct result
        :param model: result provided by model
        """
        if correct not in self.results:
            self.results[correct] = dict()
        if model not in self.results[correct]:
            self.results[correct][model] = 0
        self.results[correct][model] += 1

    def format_confusion_table(self):
        """
        Generates a formatted confusion table (for unformatted data, get model.results)
        :return: string which when printed will be a confusion matrix table
        """
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
        """
        Calculates the error rate for the model over the given test data.  If the model returns a list, each item in the
        list will be considered separately for the error rate calculation (for example, if the model returns a list of
        ten items, and ten runs each have one error in the list, the error rate will be 10%, not 100%).
        :return: Error rate in decimal format (e.g., 10% = .1)
        """
        total = 0
        error = 0
        for correct_value, model_values in self.results.items():
            for model_value, instances in model_values.items():
                total += instances
                if model_value != correct_value:
                    error += instances
        return error / total
