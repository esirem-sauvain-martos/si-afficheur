import csv

import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate, delta, prediction_type = 0):
        self._input_count = 7
        self._epoch_count = 0
        self._learning_rate = learning_rate
        self._prediction_type = prediction_type
        self._delta = delta
        self._weigths = [0 for _ in range(7)]
        self._biais = 1
        self._biais_weigth = 0
        self._error_values = []

    def predict(self, inputs):
        computed_res = 0
        for i, input_val in enumerate(inputs):
            computed_res += input_val * self._weigths[i]

        computed_res += self._biais_weigth

        if self._prediction_type == 0:
            return 0 if computed_res <= 0 else 1
        elif self._prediction_type == 1:
            return 0 if computed_res <= 0 else computed_res

    def train(self, inputs, expected_outputs):

        keep_training = True
        last_error_rate_average = 1000

        while keep_training:
            epoch_error_value = 0
            for i, segments_configuration in enumerate(inputs):
                
                prediction = self.predict(segments_configuration)

                for j, weight in enumerate(self._weigths):
                    new_weight = weight + self._learning_rate * \
                        (expected_outputs[i] - prediction) * segments_configuration[j]
                    self._weigths[j] = new_weight

                self._biais_weigth = self._biais_weigth + \
                    self._learning_rate * \
                    (expected_outputs[i] - prediction) * 1

                epoch_error_value += 0.5 * \
                    ((prediction - expected_outputs[i]) ** 2)
            self._error_values.append(epoch_error_value)

            new_error_rate_average = sum(self._error_values) / len(self._error_values)
            if abs(new_error_rate_average - last_error_rate_average) < self._delta:
                keep_training = False
            last_error_rate_average = new_error_rate_average

    def show_error_graph(self):
        plt.plot(self._error_values)
        plt.show()

    def save_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header_line = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w biais']
            csv_writer.writerow(header_line)
            csv_writer.writerow(self._weigths + [self._biais_weigth])

    def load_weight(self):
        filename = 'perceptron.csv'
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for line in csv_reader:
                weights = line
            for i in range(len(weights) - 1):
                self._weigths[i] = float(weights[i])
            self._biais_weigth = float(weights[-1])

    def __str__(self):
        return "Perceptron [weights = {}, biais weight = {}]".format(self._weigths, self._biais_weigth)
