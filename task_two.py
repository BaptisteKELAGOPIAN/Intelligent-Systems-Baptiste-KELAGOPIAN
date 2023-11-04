import numpy as np
import matplotlib.pyplot as plt

def activation_function(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return activation_function(x) * (1 - activation_function(x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.random.randn(self.hidden_size) * 0.01
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.random.randn(self.output_size) * 0.01
        
    def propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = activation_function(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.z2
        
        return self.y_hat
    
    def backpropagation(self, X, y, learning_rate):
        delta2 = self.y_hat - y
        
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.z1)
        
        self.W2 -= learning_rate * np.dot(self.a1.T, delta2)
        self.b2 -= learning_rate * np.sum(delta2, axis=0)
        self.W1 -= learning_rate * np.dot(X.T, delta1)
        self.b1 -= learning_rate * np.sum(delta1, axis=0)
        
    def train(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            y_hat = self.propagation(X)
            self.backpropagation(X, y, learning_rate)
                

X = np.linspace(0, 1, 20).reshape(-1, 1)
y = (1 + 0.6 * np.sin(2 * np.pi * X / 0.7) + 0.3 * np.sin(2 * np.pi * X)) / 2

mlp = MLP(input_size=1, hidden_size=6, output_size=1)
mlp.train(X, y, learning_rate=0.01, epochs=100000)

X_test = np.linspace(0, 1, 100).reshape(-1, 1)
y_test = (1 + 0.6 * np.sin(2 * np.pi * X_test / 0.7) + 0.3 * np.sin(2 * np.pi * X_test)) / 2
y_hat_test = mlp.propagation(X_test)

plt.plot(X_test, y_test, label="Base Function")
plt.plot(X_test, y_hat_test, label="Result Function")
plt.legend()
plt.show()