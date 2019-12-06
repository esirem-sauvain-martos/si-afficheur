import csv

from perceptron import Perceptron


def import_inputs():
    inputs = []

    with open('inputs.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                inputs.append(list(map(int, row)))

    return inputs

def write_inputs(inputs):
    with open('inputs.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        for input_row in inputs:
            csv_writer.writerow(input_row)

def main():
    outputs = range(1, 10)
    inputs = import_inputs()

    perceptron = Perceptron(0.01, 0.005, 1)
    perceptron.train(inputs, outputs)
    # perceptron.load_weight()

    for possible_input in inputs:
        print(round(perceptron.predict(possible_input)))

    perceptron.show_error_graph()

    perceptron.save_weight()



if __name__ == "__main__":
    main()
