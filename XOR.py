import numpy as np
from MLP import MLP

test_logs = open("XOR_test_hidden.txt", "w")
print("1. XOR test\n", file=test_logs)

def XOR(max_epochs, learning_rate, NH):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    NI = 2
    NO = 1
    NN = MLP(NI, NH, NO)

    NN.randomise()

    print('Epochs = ' + str(max_epochs), file=test_logs)

    print('Learning rate = ' + str(learning_rate), file=test_logs)

    print('Hidden units = ' + str(NH) + '\n\n', file=test_logs)

    print('Before Training:\n', file=test_logs)

    for i in range(len(inputs)):
        NN.forward(inputs[i], 'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=test_logs)

    print('\nTraining:\n', file=test_logs)


    # training
    for i in range(max_epochs):
        NN.forward(inputs, 'sigmoid')
        error = NN.backwards(inputs, outputs, 'sigmoid')
        NN.update_weights(learning_rate)

        if (i + 1) == 100:
            print(' Error at Epoch:\t' + str(i + 1) + '\t\t  is \t' + str(error), file=test_logs)

        elif (i + 1) == 1000 or (i + 1) == 10000 or (i + 1) == 100000 or (i + 1) == 1000000:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=test_logs)


    print('\nAfter Training :\n', file=test_logs)


    # get accuracy after training
    accuracy = float()
    for i in range(len(inputs)):
        NN.forward(inputs[i], 'sigmoid')
        print('Target:\t {}  Output:\t {}'.format(str(outputs[i]), str(NN.O)), file=test_logs)

        if (outputs[i][0] == 0):
            accuracy += 1 - NN.O[0]
        elif (outputs[i][0] == 1):
            accuracy += NN.O[0]
    print('\n\nAccuracy = ' + str(accuracy / 4), file=test_logs)


epochs = [1000000]
learning_rate = [1.0, 0.75, 0.5, 0.25, .05]
num_hidden = 7

for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        XOR(epochs[i], learning_rate[j], num_hidden)
        print('\n-----------------------------------------------------------------------------\n', file=test_logs)
