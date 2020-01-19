import csv

from perceptron import Perceptron

def import_inputs(path='inputs.csv'):
    inputs = []

    with open(path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        for index, row in enumerate(csv_reader):
            if index > 0:
                inputs.append(list(map(int, row)))

    return inputs

def test_perceptron_1(perceptron, inputs):
    results = []

    for possible_input in inputs.copy():
        inp = possible_input[:]
        inp.append(1)
        results.append(int(round(perceptron.predict(inp, 0))))

    print(results)

def test_perceptron_4(perceptron, inputs):
    results = []

    for inp in inputs:
        res = 0
        for i in range(4):
            sg = inp[:]
            sg.append(1)
            res = res << 1
            res += int(round(perceptron.predict(sg, i)))
        results.append(res)

    print(results)

def main():
    inputs = import_inputs()
    
    outputs_4 = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1]
    ]

    outputs_1 = []
    for i in range(1, 10):
        outputs_1.append([i])
    # outputs_1 = [
    #     [1],
    #     [2],
    #     [3],
    #     [4],
    #     [6],
    #     [5],
    #     [7],
    #     [8],
    #     [9],
    # ]

    perceptron_1 = Perceptron(7, 1, 0.01, 0.001, prediction_type=1)
    perceptron_4 = Perceptron(7, 4, 0.01, 0.002, prediction_type=1)

    perceptron_1.train(inputs, outputs_1)
    # perceptron_1.train(import_inputs('inputs65.csv'), outputs_1)
    perceptron_4.train(inputs, outputs_4)

    test_perceptron_1(perceptron_1, import_inputs())
    # test_perceptron_4(perceptron_4, inputs)

    perceptron_4.show_error_graph()

    # perceptron.save_weight()


if __name__ == "__main__":
    main()
