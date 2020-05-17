# Original code by Victor Zhou, with minor changes
# You can find it at: https://victorzhou.com/blog/intro-to-neural-networks/

import numpy as np
import network
import neuron

weights = np.array([0,1])
bias = 4
n = neuron.Neuron(weights, bias)

x = np.array([2,3])

print("Neuron output test:")
print(f'The neuron output is {n.feedforward(x)}', end ='\n\n')


network = network.OurNeuralNetwork()
x = np.array([2, 3])

print("Neural Network test:")
print(f'The neural network output is {network.feedforward(x)}')
