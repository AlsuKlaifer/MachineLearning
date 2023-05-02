class Adaboost:
    def __init__(self, m):
        self.m = m

    def __init_weights(self, n):
        """ initialization of input variables weights """
        pass

    def update_weights(self, targets, predictions, weights, weak_classifier_weight):
        """ update weights functions DO NOT use loops """
        pass

    def calculate_error(self, targets, predictions, weights):
        """ weak classifier error calculation DO NOT use loops """
        pass

    def calculate_classifier_weight(self, targets, predictions, weights):
        """ weak classifier weight calculation DO NOT use loops """
        pass

    def train(self, inputs, targets):
        """ train model """
        pass

    def get_predictions(self, inputs):
        """ adaboost get predictions """
        pass
