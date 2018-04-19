import numpy as np
import matplotlib
import matplotlib.pyplot as plt
data = np.loadtxt('quasar_train.csv',delimiter=',')
xf=data[0,:]
xf=xf.reshape((450,1))
row=1
y=data[row,:]
y=y.reshape((450,1))
x = np.insert(xf, 0, 1, axis=1)
theta=np.zeros((2,1))
xt=np.transpose(x)
xd=np.dot(xt,x)
a=np.linalg.inv(xd)
b=np.dot(xt,y)
theta=np.dot(a,b)
print("theta from linear regression : ")
print(theta)
plot_y=np.dot(x,theta)
plt.title("Plot : row" + str(row) + " 5b(i)")
plt.scatter(range(1150, 1600, 1), y, label='Raw data')
plt.plot(range(1150, 1600, 1), plot_y, label='Linear regression line', color='red')
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.show()
t=5
def W(X):
    w=np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
        w[i,i]=np.exp(-(X-x[i,1])**2/(2*(t**2)))
    return w
'''def Theta(w):
    xw = np.dot(xt, w)
    a = np.linalg.inv(np.dot(xw, x))
    b = np.dot(xw, y)
    theta = np.dot(a, b)
    plot_y = np.dot(x, theta)
    return plot_y'''
Y=np.zeros((450,1))
for X in range(1150, 1600, 1):
    xw = np.dot(xt, W(X))
    a = np.linalg.inv(np.dot(xw, x))
    b = np.dot(xw, y)
    theta = np.dot(a, b)
    Y[(X-1150), 0] = np.dot(x, theta)
plt.title("Plot : row" + str(row) + " 5b(ii)")
plt.scatter(range(1150, 1600, 1), y, label='Raw data')
plt.plot(range(1150, 1600, 1), Y, label='Locally weighed linear regression curve', color='red')
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.show()
