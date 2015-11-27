class ConfusionMatrix:
    def __init__(self, function):
        self.function = function

    def build(self, test_data):
        confusions = dict()
        for input, correct_value in test_data:
            model_value = self.function(input)
            if model_value is list:
                for correct, model in zip(correct_value, model_value):
                    self.evaluate_and_record(confusions, correct, model)
            else:  # scalar
                self.evaluate_and_record(confusions, correct_value, model_value)

    def evaluate_and_record(self, confusions, correct, model):
        if correct == model:
            return
        if correct not in confusions:
            confusions[correct] = dict()
        if model not in confusions[correct]:
            confusions[correct][model] = 0
        confusions[correct][model] += 1
