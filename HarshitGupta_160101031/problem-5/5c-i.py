import numpy as np
import matplotlib.pyplot as plt

""" File to smoothen the data """

tau = 5

""" Function to create the diagonal Matrix W for locally weighted regression """


def make_W(X, x):
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[0]):
        # print(-(x-X[i,1])**2/(2*(tau**2)))
        W[i][i] = np.exp(-(x - X[i, 1]) ** 2 / (2 * (tau ** 2)))
    return W


# formula for theta using normal equations is theta = inverse(X'WX)X'Wy
def get_theta(X, Y, x):
    W = make_W(X, x)
    temp = np.matmul(np.transpose(X), W)
    temp = np.matmul(temp, X)
    temp = np.linalg.inv(temp)
    temp2 = np.matmul(W, Y)
    temp2 = np.matmul(np.transpose(X), temp2)
    return np.matmul(temp, temp2)


#run the file once for quasar_train and then for quasar_test

# frequency range from 1150 to 1599
data = np.loadtxt('quasar_test.csv', delimiter=',', skiprows=1)
# data = np.loadtxt('problem-5/quasar_train.csv', delimiter=',', skiprows=1)

print(data.shape)
data_new = np.zeros((1,450))

# record number for which the plot is to be generated
x_axis = range(1150, 1600, 1)
X = np.array(range(1150, 1600, 1))
X = X.reshape(450, 1)
X = np.hstack((np.zeros((450, 1)) + 1, X))

for item_number in range(0,50):

    print("processing item number "+str(item_number))
    y_axis = data[item_number, :]

    """ Formatting the data  , adding the intercept term """

    Y = y_axis.reshape(450, 1)

    y_locally_weighted_regression = []
    for x in x_axis:
        theta2 = get_theta(X, Y, x)
        y_locally_weighted_regression.append(theta2[1] * x + theta2[0])
   
    data_new = np.vstack((data_new,np.array(y_locally_weighted_regression).reshape(1,450)))
   

# data_new = np.array(data_new)
data_new = data_new[1:,]
print(data_new.shape)
# print(data_new)
np.savetxt('new_test.csv',data_new,delimiter=',')
# np.savetxt('new_train.csv',data_new,delimiter=',')
