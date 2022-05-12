import random
import numpy as np
from math import exp


class Network:
    in_layer = []

    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        self.weights = []
        self.input_size = input_size
        self.output_size = output_size
        self.output = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.max_error = max_error
        self.bias = bias
        self.net = []
        self.current_error_signal = []
        self.randomize_weights()

    def randomize_weights(self):
        for idx in range(self.input_size + 1):
            self.weights.append([random.uniform(-1, 1) for idx in range(self.output_size)])

    def compute_net(self, input_layer):
        net_value_holder = 0
        self.net.clear()
        for out_idx in range(self.output_size):
            for in_idx in range(len(input_layer)):
                net_value_holder += input_layer[in_idx] * self.weights[in_idx][out_idx]
            net_value_holder += self.bias * self.weights[len(input_layer)][out_idx]
            self.net.append(net_value_holder)

    def compute_outputs(self, input_layer):
        self.in_layer = input_layer
        self.compute_net(input_layer)
        for out_idx in range(self.output_size):
            self.output[out_idx] = self.activation(self.net[out_idx])

    # relu is default activation
    def activation(self, value):
        return 1 if value >= 0 else 0

    def activation_derivative(self, value):
        return 1 if value > 0 else 0

    def propagate_input(self, input_layer):
        self.compute_outputs(input_layer)

    def update_weights(self, l_rate, errs):
        for idx1 in range(self.output_size):
            for idx2 in range(self.input_size):
                self.weights[idx2][idx1] += l_rate * self.in_layer[idx2] * errs[idx1]
            self.weights[self.input_size][idx1] += l_rate * self.bias * errs[idx1]

    def comp_error_signals(self, prev_errs, prev_layer_weights):
        we_sum = 0
        self.current_error_signal.clear()
        for idx1 in range(len(prev_layer_weights) - 1):
            for idx2 in range(len(prev_errs)):
                we_sum += prev_layer_weights[idx1][idx2] * prev_errs[idx2]
            self.current_error_signal.append(self.activation_derivative(self.net[idx1]) * we_sum)


class Linear(Network):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0, a_factor=1):
        Network.__init__(self, input_size, output_size, learning_rate, max_error, bias)
        self.a_factor = a_factor

    def activation(self, value):
        return self.a_factor * value


class Sigmoid(Network):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        Network.__init__(self, input_size, output_size, learning_rate, max_error, bias)

    def activation(self, value):
        if abs(value) >= 710:
            return 0
        return 1 / (1 - exp(value * (-1)))

    def activation_derivative(self, value):
        return self.activation(value) * (1 - self.activation(value))


class ReLu(Network):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0):
        Network.__init__(self, input_size, output_size, learning_rate, max_error, bias)

    def activation(self, value):
        return 1 if value >= 0 else 0


class UniSigmoid(Network):
    def __init__(self, input_size, output_size, learning_rate=3, max_error=0.01, bias=0, l_factor=1):
        Network.__init__(self, input_size, output_size, learning_rate, max_error, bias)
        self.l_factor = l_factor

    def activation(self, value):
        if abs(value) >= 710:
            return 0
        return 1 / (1 + exp(-value * self.l_factor))

    def activation_derivative(self, value):
        return (self.l_factor * exp(-self.l_factor * value)) / pow(exp(-self.l_factor * value) + 1, 2)


class NN:
    output_layer_errs = []

    def __init__(self, input_size, output_size, learning_rate=0.5):
        self.start_l_rate = learning_rate
        self.learning_rate = learning_rate
        self.first_layer = Sigmoid(input_size, 3, bias=1)
        self.final_layer = UniSigmoid(3, output_size, bias=1, l_factor=2)
        self.error = 0

    def propagate(self, input_layer):
        self.first_layer.propagate_input(input_layer)
        self.final_layer.propagate_input(self.first_layer.output)

    def compute_error(self, targets):
        self.comp_output_layer_errs(targets)
        self.error = 0
        for idx in range(len(self.final_layer.output)):
            self.error += pow(targets[idx] - self.final_layer.output[idx], 2) / 2

    def comp_output_layer_errs(self, targets):
        error = 0.0
        self.output_layer_errs.clear()
        for idx in range(self.final_layer.output_size):
            error = targets[idx] - self.final_layer.output[idx]
            error *= self.final_layer.activation_derivative(self.final_layer.net[idx])
            self.output_layer_errs.append(error)

    # Compute all layers errors first, then update weights.
    def back_propagate(self):
        if self.learning_rate < 0.01:
            self.learning_rate = self.start_l_rate
        self.first_layer.comp_error_signals(self.output_layer_errs, self.final_layer.weights)
        self.first_layer.update_weights(self.learning_rate, self.first_layer.current_error_signal)
        self.final_layer.update_weights(self.learning_rate, self.output_layer_errs)
        self.learning_rate *= 0.9

    def reset(self):
        self.first_layer = UniSigmoid(self.first_layer.input_size, 5, bias=1, l_factor=1)
        self.final_layer = UniSigmoid(5, self.final_layer.output_size, bias=1, l_factor=1)


in_values = [[-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]
in_targets = [0, 1, 1, 0]
max_err = 0.01

net = NN(3, 1, learning_rate=0.5)
e = 10
counter = 0
while e > max_err:
    e = 0
    for i in range(len(in_values)):
        net.propagate(in_values[i])
        net.compute_error([in_targets[i]])
        e += net.error
        net.back_propagate()
    # if counter % 10000 == 0:
    print("error: ", e)

for i in range(len(in_values)):
    net.propagate(in_values[i])
    print(net.final_layer.output)
