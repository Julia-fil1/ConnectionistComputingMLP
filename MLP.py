import numpy as np


class MLP:
    def __init__(self, NI, NH, NO):
        self.no_in = NI
        self.no_hidden = NH
        self.no_out = NO
        self.W1 = np.array
        self.W2 = np.array
        self.dW1 = np.array
        self.dW2 = np.array
        self.Z1 = np.array
        self.Z2 = np.array
        self.H = np.array
        self.O = np.array

    def randomise(self):
        # initializing W1 and W2 to small random values, range from 0 to 1, for both layers from input to hidden unit
        # and from hidden units to outputs
        self.W1 = np.array((np.random.uniform(0, 1, (self.no_in, self.no_hidden))).tolist())
        self.W2 = np.array((np.random.uniform(0, 1, (self.no_hidden, self.no_out))).tolist())

        # set dW2 and dW2 to all zeroes
        self.dW1 = np.dot(self.W1, 0)
        self.dW2 = np.dot(self.W2, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_sigmoid(self, x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    def tanh(self, x):
        return (2 / (1 + np.exp(x * -2))) - 1

    def derivative_tanh(self, x):
        return 1 - (np.power(self.tanh(x), 2))

    def forward(self, I, activation):
        if activation == 'sigmoid':
            self.Z1 = np.dot(I, self.W1)
            self.H = self.sigmoid(self.Z1)
            self.Z2 = np.dot(self.H, self.W2)
            self.O = self.sigmoid(self.Z2)

        elif activation == 'tanh':
            self.Z1 = np.dot(I, self.W1)
            self.H = self.tanh(self.Z1)
            self.Z2 = np.dot(self.H, self.W2)
            self.O = self.Z2

        return self.O

    def backwards(self, I, target, activation):
        output_error = np.subtract(target, self.O)
        if activation == 'sigmoid':
            activation_upper = self.derivative_sigmoid(self.Z2)
            activation_lower = self.derivative_sigmoid(self.Z1)
        elif activation == 'tanh':
            activation_upper = self.derivative_tanh(self.Z2)
            activation_lower = self.derivative_tanh(self.Z1)
        dw2_a = np.multiply(output_error, activation_upper)
        self.dW2 = np.dot(self.H.T, dw2_a)
        dw1_a = np.multiply(np.dot(dw2_a, self.W2.T), activation_lower)
        self.dW1 = np.dot(I.T, dw1_a)
        return np.mean(np.abs(output_error))

    def update_weights(self, learning_rate):
        self.W1 = np.add(self.W1, learning_rate * self.dW1)
        self.W2 = np.add(self.W2, learning_rate * self.dW2)
        self.dW1 = np.array
        self.dW2 = np.array
