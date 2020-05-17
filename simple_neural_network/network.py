# Original code by Victor Zhou, with minor changes
# You can find it at: https://victorzhou.com/blog/intro-to-neural-networks/

import numpy as np
import neuron

class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = neuron.Neuron(weights, bias)
        self.h2 = neuron.Neuron(weights, bias)
        self.o1 = neuron.Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1