import numpy as np
import matplotlib.pyplot as plt


# distance for the f_right

def distance(f1, f2):
    x1 = f1[150:450]
    x2 = f2[150:450]
    d = np.sum(np.power(np.subtract(x1, x2), 2))
    return d


def ker(x):
    return max(0, 1 - x)


""" function to to estimate on one spectrum """


def estimate(x_train, x_test, id):
    test_example = x_test[id]
    predictions = []
    m_train = x_train.shape[0]
    distances = np.zeros((m_train, 1))

    for i in range(0, m_train):
        distances[i, 0] = distance(x_train[i], test_example)

    h = np.max(distances)
    min_indices = np.argsort(distances, axis=0)[0:3]

    ker_dis_by_h = np.zeros((3, 1))

    for i in range(0, 3):
        ker_dis_by_h[i] = ker(np.divide(distances[min_indices[i]], h))
    s = np.sum(ker_dis_by_h)

    for i in range(0, 50):
        num = np.sum(np.multiply(ker_dis_by_h, x_train[min_indices, i]))
        predictions.append(num / s)

    return predictions


def avg_error(x_train, x_test):
    m_test = x_test.shape[0]
    error = 0
    for i in range(0, m_test):
        prediction = estimate(x_train, x_test, id=i)
        error = error + sum((prediction - data_test[i, 0:50]) ** 2)
    return error/m_test


data_train = np.loadtxt('new_train.csv', delimiter=',', skiprows=0)
data_test = np.loadtxt('new_test.csv', delimiter=',', skiprows=0)

print(data_train.shape)
print(data_test.shape)

x_pred = range(1150, 1200, 1)
x_axis = range(1150, 1600, 1)

# error = sum((predict - data_test[id, 0:50]) ** 2)
# print("error : " + str(error))

id = 0  # test example 1
predict = estimate(data_train, data_test, id=id)
plt.figure(0)
plt.plot(x_axis, data_test[id], label='training data')
plt.plot(x_pred, predict, label='predicted curve', color='red')
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("Estimation of f-left on example 1")
plt.legend()
plt.savefig("images/5c-iii-a")
plt.show()

id = 5  # test example 6
predict = estimate(data_train, data_test, id=id)
plt.figure(1)
plt.plot(x_axis, data_test[id], label='training data')
plt.plot(x_pred, predict, label='predicted curve', color='red')
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.title("Estimation of f-left on example 6")
plt.legend()
plt.savefig("images/5c-iii-b")
plt.show()


print("average error = "+str(avg_error(data_train,data_test)))