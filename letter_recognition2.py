import numpy as np
import pandas as pd
from MLP import MLP

test_logs = open("LETTER_test_hidden.txt", "w")
print("Q3 LETTER TEST\n", file=test_logs)


def letter(max_epochs, learning_rate, num_hidden):
    np.random.seed(1)

    inputs = []
    outputs = []
    data_output = []
    columns = ["letter", "x-box", "y-box", "width", "height", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
               "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    df = pd.read_csv("letter-recognition.data", names=columns)
    data_output = df["letter"]

    for i in range(len(data_output)):
        outputs.append(ord(str(data_output[i])) - ord('A'))

    inputs = df.drop(["letter"], axis=1)
    inputs = np.array(inputs)
    inputs = inputs / 15  # normalization

    # train set
    inputs_train = inputs[:16000]
    categorical_y = np.zeros((16000, 26))
    for i, l in enumerate(outputs[:16000]):
        categorical_y[i][l] = 1
    outputs_train = categorical_y

    # test set
    inputs_test = inputs[16000:]
    num_inputs = 16
    num_outputs = 26

    NN = MLP(num_inputs, num_hidden, num_outputs)
    NN.randomise()
    print('Epochs = ' + str(max_epochs), file=test_logs)

    print('Learning rate = ' + str(learning_rate), file=test_logs)

    print('Hidden units = ' + str(num_hidden) + '\n\n', file=test_logs)

    for i in range(0, max_epochs):
        NN.forward(inputs_train, 'tanh')
        error = NN.backwards(inputs_train, outputs_train, 'tanh')
        NN.update_weights(learning_rate)

        if (i + 1) % (max_epochs / 20) == 0:
            print(' Error at Epoch:\t' + str(i + 1) + '\t  is \t' + str(error), file=test_logs)

    # testing
    def to_character0(output_vector):
        list_of_vectors = list(output_vector)
        a = list_of_vectors.index(max(list_of_vectors))
        return chr(a + ord('A'))

    prediction = []
    for i in range(4000):
        NN.forward(inputs_test[i], 'tanh')
        prediction.append(to_character0(NN.O))

    def to_character(n):
        return chr(int(n) + ord('A'))

    correct_letter = {to_character(i): 0 for i in range(26)}
    letter_num = {to_character(i): 0 for i in range(26)}

    print('\n-----------------------------------------------------------------------------\n', file=test_logs)

    for i, _ in enumerate(data_output[16000:]):
        letter_num[data_output[16000 + i]] += 1
        # predictions
        if i % 300 == 0:
            print('Expected: {} | Output: {}'.format(data_output[16000 + i], prediction[i]), file=test_logs)
        if data_output[16000 + i] == prediction[i]:
            correct_letter[prediction[i]] += 1

    print('\n-----------------------------------------------------------------------------\n', file=test_logs)

    # Calculate the accuracy
    accuracy = sum(correct_letter.values()) / len(prediction)
    print('Test sample size: {} | Correctly predicted sample size: {}'.format(len(prediction),
                                                                              sum(correct_letter.values())),
          file=test_logs)
    print('Accuracy: %.3f' % accuracy, file=test_logs)

    # Performance of prediction of each letter
    print('\n-----------------------------------------------------------------------------\n', file=test_logs)

    for k, v in letter_num.items():
        print('{} => Number of occurrences in the sample: {}'
              ' | Number of correct predictions: {}'
              ' | Accuracy: {}'.format(k, v, correct_letter[k], correct_letter[k] / v),
              file=test_logs)


epochs = [1000000]
learning_rate = [0.000005]
num_hidden = 10
for i in range(len(epochs)):
    for j in range(len(learning_rate)):
        print('-----------------------------------------------------------------------------\n', file=test_logs)
        letter(epochs[i], learning_rate[j], num_hidden)
        print('\n-----------------------------------------------------------------------------\n', file=test_logs)
