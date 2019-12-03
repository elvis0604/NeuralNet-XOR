import numpy as numpy
import matplotlib.pyplot as plot

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

#backprop utility
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

#forward func
def forward(x, w1, w2, predict = False):
    a1 = numpy.matmul(x, w1)
    z1 = sigmoid(a1)

    #make and add bias
    bias = numpy.ones((len(z1), 1))
    z1 = numpy.concatenate((bias, z1), axis = 1)
    a2 = numpy.matmul(z1, w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1, z1, a2, z2

#backprop func
def backprop(a2, z0, z1, z2, y):
    delta2 = z2  - y
    Delta2 = numpy.matmul(z1.T, delta2)
    delta1 = (delta2.dot(w2[1:,:].T)) * sigmoid_der(a1)
    Delta1 = numpy.matmul(z0.T, delta1)
    return delta2, Delta1, Delta2

#first column is bias
x = numpy.array([[1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]])

#expect output
y = numpy.array([[0], [1], [1], [0]])

#random weight
w1 = numpy.random.randn(3, 5)
w2 = numpy.random.randn(6, 1)

#learning rate
lr = 0.1
costs = []
epochs = 15000
m = len(x)

#start
for i in range(epochs):
    #forward
    a1, z1, a2, z2 = forward(x, w1, w2)
    
    #backprop
    delta2, Delta1, Delta2 = backprop(a2, x, z1, z2, y)
    
    w1 -= lr * (1 / m) * Delta1
    w2 -= lr * (1 / m) * Delta2

    #add cost
    c = numpy.mean(numpy.abs(delta2))
    costs.append(c)
    
    if i % 1000 == 0: #show less iteration
        print(f"Iteration: {i} Error: {c}")

#prediction
z3 = forward(x, w1, w2, True)
print("Percentage: ")
print(z3)
print("Prediction: ")
print(numpy.round(z3))

#plotting
plot.plot(costs)
plot.show()