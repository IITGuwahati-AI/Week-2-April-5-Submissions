import numpy as np
import matplotlib.pyplot as plt


""" Function to create the diagonal Matrix W for locally weighted regression """


def make_W(X, x,tau):
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[0]):
        # print(-(x-X[i,1])**2/(2*(tau**2)))
        W[i][i] = np.exp(-(x - X[i, 1]) ** 2 / (2 * (tau ** 2)))
    return W


# formula for theta using normal equations is theta = inverse(X'WX)X'Wy
def get_theta(X, Y, x,tau):
    W = make_W(X, x,tau)
    temp = np.matmul(np.transpose(X), W)
    temp = np.matmul(temp, X)
    temp = np.linalg.inv(temp)
    temp2 = np.matmul(W, Y)
    temp2 = np.matmul(np.transpose(X), temp2)
    return np.matmul(temp, temp2)


# frequency range from 1150 to 1599
data = np.loadtxt('quasar_train.csv', delimiter=',', skiprows=1)
print(data.shape)

# record number for which the plot is to be generated
item_number = 1

y_axis = data[item_number, :]
x_axis = range(1150, 1600, 1)

print(y_axis.shape)

""" Formatting the data  , adding the intercept term """

X = np.array(range(1150, 1600, 1))
X = X.reshape(450, 1)
X = np.hstack((np.zeros((450, 1)) + 1, X))
Y = y_axis.reshape(450, 1)

print(X.shape)

""" Calculating theta for Linear Regression using Normal equations """

Xtrans_X = np.matmul(np.transpose(X), X)
inverse_Xtrans_X = np.linalg.inv(Xtrans_X)
Xtrans_Y = np.matmul(np.transpose(X), Y)
theta = np.matmul(inverse_Xtrans_X, Xtrans_Y)

print("theta from linear regression : ")
print(theta)

""" Plotting Linear regression Line  """

plt.figure(0)
plt.title("Plot : data entry " + str(item_number) + " 5b(i)")
plt.scatter(x_axis, y_axis, label='Original Data')
plt.plot(x_axis, theta[1] * x_axis + theta[0], label='Linear regression Line',color='red')
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.savefig("images/5b-i")
plt.show()

""" Plotting Locally Weighted regression Line """

for index,tau in enumerate([1,5,10,100,1000]):
    y_locally_weighted_regression = []
    for x in x_axis:
        theta2 = get_theta(X, Y, x,tau=tau)
        y_locally_weighted_regression.append(theta2[1] * x + theta2[0])

    plt.figure(index)
    plt.title("Plot : data entry " + str(item_number) + " 5b(ii)")
    plt.scatter(x_axis, y_axis, label='Original Data')
    plt.plot(x_axis, y_locally_weighted_regression, label='Locally weighted Regression Curve',color='red')
    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    plt.text(1500,10,"$\\tau$ (tau) = "+str(tau))
    plt.legend()
    plt.savefig("images/5b-ii_"+str(index))
    plt.show()
