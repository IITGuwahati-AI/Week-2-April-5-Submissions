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
h=np.zeros(m)
for i in range(m):
    h[i]=np.dot(x[i],theta)
maxiters=10
grad = np.zeros((n+1, 1))
hess=np.zeros((n+1,n+1))
for k in range(maxiters):
    for i in range(m):
        xt=np.transpose(x[i,:])
        xt=xt.reshape(n+1,1)
        grad -= (1/m)*(1-g(y[i]*h[i]))*y[i]*xt
        #grad=grad.reshape(n+1,1)
        xi=np.dot(xt,x[i,:])
        xi=xi.reshape(n+1,n+1)
        hess -= (1/m)*g(y[i]*h[i]) * (1-g(y[i]*h[i]))*xi
        #hess=hess.reshape(n+1,n+1)
    theta=theta-np.dot(np.linalg.pinv(hess),grad)
    #theta=theta.reshape(n+1,1)
    print (theta)
