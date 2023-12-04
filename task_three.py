import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, c, r):
    return np.exp(-(x - c)**2 / (2 * r**2))

x = np.arange(0.1, 1.05, 1/22) #input

y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7) + 0.3 * np.sin(2 * np.pi * x)) / 2 #desired output

c1 = 0.5
c2 = 0.3
r1 = 0.2
r2 = 0.3

rbf1 = gaussian(x, c1, r1) #rbf activations 
rbf2 = gaussian(x, c2, r2)

X = np.column_stack((rbf1, rbf2, np.ones_like(x))) #input matrix

w = np.random.rand(3) #random weigts

epochs = 100000
learning_rate = 0.1

for _ in range(epochs):
    print(_)
    for i in range(len(x)):
        output = np.dot(w, X[i])
        error = y[i] - output
        w += learning_rate * error * X[i]

print("Final weights:", w)
predicted_outputs = np.dot(X, w)

plt.plot(x, y, label='Actual')
plt.plot(x, predicted_outputs, label='Predicted')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Actual vs Predicted Outputs')
plt.legend()
plt.show()
