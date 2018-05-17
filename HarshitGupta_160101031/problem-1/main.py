import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import time

""" Sigmoid function """


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


""" decision function """


def decide(x, theta):
    h = np.matmul(x, theta)
    h = sigmoid(h[0])
    if h >= 0.5:
        return 1
    else:
        return 0


""" Importing the dataset """

data = np.loadtxt('logistic_x.txt')
Y = np.loadtxt('logistic_y.txt')
Y = np.array([0 if x == -1 else 1 for x in Y])
data_1 = np.array([x for i, x in enumerate(data) if Y[i] == 1])
data_2 = np.array([x for i, x in enumerate(data) if Y[i] == 0])
Y = Y.reshape(99, 1)

# print(Y)

print("The dimensions of input dataset is " + str(data.shape))
print("The dimensions of output dataset is " + str(Y.shape))

""" Plotting the raw data """

plt.plot(data_1[:, 0], data_1[:, 1], 'gX', )
plt.plot(data_2[:, 0], data_2[:, 1], 'bo')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

""" Creating the input matrix , adding the intercept term """

X = np.hstack((np.zeros((data.shape[0], 1)) + 1, data))
theta = np.zeros((3, 1))
alpha = 0.01
m = data.shape[0]
n = data.shape[1]
# print(X.shape)
# print(theta.shape)
# print(Y.shape)


# """ Batch Gradient Descent """
# for i in range(1, 10000):
#     h = sigmoid(np.matmul(X, theta))
#     temp = h - Y
#     theta = theta - (alpha / m) * (np.matmul(np.transpose(X), temp))

""" Newtons method of minimisation """

for _ in range(1, 15):  # since it converges much earlier 15 iterations are more than enough

    H = np.zeros((n + 1, n + 1))  # H : Hessian matrix
    h = sigmoid(np.matmul(X, theta))  # h : hypothesis function
    temp = h - Y
    h = h * (1 - h)
    for i in range(0, m):  # always check the dimensions of the matrix after slicing
        temp_xi = X[i]
        temp_xi = temp_xi.reshape(1, n + 1)
        H = H + h[i] * (1 - h[i]) * np.matmul(np.transpose(temp_xi), temp_xi)
    H = H / m
    # print(H.shape)
    grad = (np.matmul(np.transpose(X), temp)) / m
    theta = theta - np.matmul(np.linalg.pinv(H), grad)


print("Theta : ")
print(theta)

""" Printing the Region """


m = -(theta[1] / theta[2])
c = -(theta[0] / theta[2])
line_x = np.arange(plt.xlim()[0], plt.xlim()[1], 0.1)
line_y = np.multiply(line_x, m) + c
plt.title("Decision Boundary using Newtons Method ")
plt.fill_between(line_x, plt.ylim()[0], line_y, color='#dba55e')
plt.fill_between(line_x, plt.ylim()[1], line_y, color='#d8d0be')
plt.savefig("Problem-1")
plt.show()
