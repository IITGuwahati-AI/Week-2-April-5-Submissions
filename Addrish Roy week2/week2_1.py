import numpy as np
import matplotlib
import matplotlib.pyplot as plt
xf = np.loadtxt('logistic_x.txt')
y = np.loadtxt('logistic_y.txt')
colors = ['red', 'blue']
plt.scatter(xf[:, 0], xf[:, 1], c=y,cmap=matplotlib.colors.ListedColormap(colors))
plt.xlabel('x1')
plt.ylabel('x2')
m=y.shape[0]
n=xf.shape[1]
theta=np.zeros((n+1,1))
print (theta)
x=np.insert(xf,0,1,axis=1)
print (x)
print (y)
def g(t):
    return 1/(1+np.exp(-t))
def h(a):
    return np.dot(a,theta)
def costj(theta):
    l=0
    for i in range(m):
        u = y[i]*h(x[i, :])
        l-=np.log(u)/m
    return l
def gradient(l):
    grad=0
    for i in range(m):
        grad -= (1-g(y[i]*h(x[i, :])))*y[i]*x[i, :]/m
        return grad
def hessian(l):
    hess=np.zeros((n+1,n+1))
    for i in range(m):
        xi=np.dot(x[i,:],np.transpose(x[i,:]))
        hess -= g(y[i]*h(x[i, :])) * (1-g(y[i]*h(x[i, :])))*xi/m
        return hess
maxiters=10
for k in range(maxiters):
    theta=theta-np.matmul(np.linalg.pinv(hessian(l=costj(theta))),gradient(l=costj(theta)))
    print (theta)
p=np.zeros(m,1)
plot_y=np.zeros(m,1)
for i in range(m):
    p[i]= g(y[i]*h(x[i, :]))
    if p>0.5:
        plot_y[i]=1
    else:
        plot_y[i]=-1

plt.plot(xf,plot_y)
plt.show()
