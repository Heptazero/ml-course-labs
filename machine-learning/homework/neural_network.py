import numpy as np

def get_accuracy(y, z):
    I = (y == z).astype(int)
    return np.average(I)

def softmax(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 2 and v.shape[1] > 1:
        m = np.max(v, axis=1, keepdims=True)
        e = np.exp(v - m)
        return e / np.sum(e, axis=1, keepdims=True)
    m = np.max(v, axis=0, keepdims=True)
    e = np.exp(v - m)
    return e / np.sum(e, axis=0, keepdims=True)

class IdentityActivator:
    def value(self, s):
        return s

    def derivative(self, s):
        return 1

class SigmoidActivator:
    def value(self, x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        y = self.value(x)
        return y * (1 - y)

class ReLUActivator:
    def value(self, s):
        return np.maximum(0, s)

    def derivative(self, s):
        return (s > 0).astype(float)

class SoftmaxCrossEntropy:
    def value(self, y, v):
        p = np.clip(softmax(v), 1e-12, 1.0)
        return - (y * np.log(p)).sum()

    def derivative(self, y, v):
        p = softmax(v)
        return p - y

class MSE:
    def value(self, y, v):
        return (v - y) ** 2

    def derivative(self, y, v):
        return 2 * (v - y)

class Layer:
    def __init__(self, n_input, n_output, activator=IdentityActivator()):
        self.activator = activator
        r = np.sqrt(6.0 / (n_input + n_output))
        self.W = np.random.uniform(-r, r, (n_output, n_input))
        self.b = np.zeros((n_output, 1))
        self.outputs = np.zeros((n_output, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.sums = self.W.dot(inputs) + self.b
        self.outputs = self.activator.value(self.sums)

    def back_propagation(self, delta_in, learning_rate):
        d = self.activator.derivative(self.sums) * delta_in
        self.delta_out = self.W.T.dot(d)
        self.W -= learning_rate * d.dot(self.inputs.T)
        self.b -= learning_rate * d

class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x):
        inputs = x
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs
        return inputs

    def back_propagation(self, y, outputs, learning_rate):
        delta_in = self.loss.derivative(y, outputs)
        for layer in self.layers[::-1]:
            layer.back_propagation(delta_in, learning_rate)
            delta_in = layer.delta_out

    def fit(self, X, y, N, learning_rate):
        for t in range(N):
            i = np.random.randint(0, len(X))
            outputs = self.forward(X[i].reshape(-1, 1))
            self.back_propagation(y[i].reshape(-1, 1), outputs, learning_rate)

    def predict(self, X):
        y = []
        for i in range(len(X)):
            p = self.forward(X[i].reshape(-1, 1)).reshape(-1)
            y.append(p)
        return np.array(y)
