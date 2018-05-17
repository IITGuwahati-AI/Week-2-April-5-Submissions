import numpy as np
import matplotlib
import matplotlib.pyplot as plt
xf = np.loadtxt('logistic_x.txt')
y = np.loadtxt('logistic_y.txt')
m=y.shape[0]
n=xf.shape[1]
theta=np.zeros((n+1,1))
#print (theta)
x=np.insert(xf,0,1,axis=1)
#print (x)
#print (y.shape)
#y=y.reshape(m,1)
def g(t):
    return 1/(1+np.exp(-t))
h=np.zeros((m,1))
#print (h.shape)
h=np.dot(x,theta)
maxiters=5
grad = np.zeros((n+1, 1))
#print(grad.shape)
hess=np.zeros((n+1,n+1))
for k in range(maxiters):
    for i in range(m):
        xu = x[i].reshape(1, n+1)
        xt=np.transpose(xu)
        #xt=xt.reshape(n+1,1)
        grad -= (1/m)*(1-g(y[i]*h[i]))*y[i]*xt
        #grad=grad.reshape(n+1,1)
        xi=np.dot(xt,xu)
        #xi=xi.reshape(n+1,n+1)
        hess -= (1/m)*g(y[i]*h[i]) * (1-g(y[i]*h[i]))*xi
        #hess=hess.reshape(n+1,n+1)
        hinv = np.linalg.pinv(hess)
    theta=theta-np.dot(hinv,grad)
    #theta=theta.reshape(n+1,1)
print (theta)
h = np.dot(x, theta)
#print (h)
x1 = xf[:, 0]
x2 = -(theta[0] + theta[1]*x1)/theta[2]
colors = ['red', 'blue']
s=['-1','+1']
plt.scatter(xf[:, 0], xf[:, 1], c=y,cmap=matplotlib.colors.ListedColormap(colors),label=s)
plt.legend()
plt.title("Decision Boundary for Logistic Regression using Newtons Method ")
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x1,x2)
plt.show()
