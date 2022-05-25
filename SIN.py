import numpy as np
from MLP import MLP

test_logs = open("SIN_test_hidden.txt", "w")
print("Q2 SIN_test\n", file=test_logs)


def SIN(max_epochs, learning_rate, NH):
    inputs = []
    outputs = []

    print('Epochs = ' + str(max_epochs), file=test_logs)

    print('Learning rate = ' + str(learning_rate), file=test_logs)

    print('Hidden units = ' + str(NH) + '\n\n', file=test_logs)

    print('Before Training:\n', file=test_logs)

    for i in range(500):
        vector = list(np.random.uniform(-1.0, 1.0, 4))
        vector = [float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])]

        inputs.append(vector)

    inputs = np.array(inputs)

    for i in range(500):
        outputs.append(np.sin([inputs[i][0] - inputs[i][1] + inputs[i][2] - inputs[i][3]]))

    num_inputs = 4
    num_outputs = 1
    NN = MLP(num_inputs, NH, num_outputs)

    NN.randomise()

    for i in range(400):
        NN.forward(inputs[i], 'sigmoid')
        print(' Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=test_logs)

    print('\nTraining:\n', file=test_logs)


    # training
    for i in range(max_epochs):
        NN.forward(inputs[:400], 'tanh')
        error = NN.backwards(inputs[:400], outputs[:400], 'tanh')
        NN.update_weights(learning_rate)

        if (i + 1) == 100:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t' + str(error), file=test_logs)

        elif (i + 1) == 1000 or (i + 1) == 10000 or (i + 1) == 100000 or (i + 1) == 1000000:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=test_logs)

    print('\nAfter Training :\n', file=test_logs)

    # testing
    diff = 0
    for i in range(400, len(inputs)):
        NN.forward(inputs[i], 'tanh')
        print(' Target:\t{}\t Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=test_logs)
        diff = diff + np.abs(outputs[i][0] - NN.O[0])

    accuracy = 1-(diff/100)
    print('\nAccuracy = ' + str(accuracy), file=test_logs)


epochs = [1000000]
learning_rate = [0.1, 0.01, 0.001, 0.0001]
num_hidden = 40

for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        SIN(epochs[i], learning_rate[j], num_hidden)
        print('\n-----------------------------------------------------------------------------------\n', file=test_logs)
