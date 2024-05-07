# %%
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
%config inlinebackend.figure_format='retina'

x = np.arange(10).reshape(-1,1)
y = ((20) 
     * x ** 2
     + 50
    # + (np.random.randint(0, 20, size=10)
      #           .reshape(-1,1))
    )
w = np.array([[2]])
b = np.array([0])
x.shape, y.shape, w.shape

def plot_model(x, y, model, plot=None):
    if plot:
        fig, ax = plot
    else:
        fig, ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x, y, label='data')
    ax.scatter(x, model.forward(x), label='prediction', marker='+')
   # ax.axline((0, model.b[0]), slope=model.w[0,0], color='C2', label='model')
    fig.legend()
    fig.show()
    return (fig, ax)
"""
a = 1.0
b_int = 0.0
w = np.array([[a]])
b = np.array([b_int])
model = Model(w, b)
plot = plot_model(x, y, model)
"""

def abs(x):
    return np.abs(x)

def mean(x):
    return 

def mae(y, x, w, b):
    return (1 / len(x)) * np.sum(abs((x * w[:, [0]] + b) - y))

mae(y, x, w, b)


def abs_d(x):
    x = x.copy()
    x[x < 0] = -1
    x[x >= 0] = 1
    return x
    #return x / (np.abs(x) + 0.00001)

def mae_d(y, x, w, b, model):
    y_predict = model.forward(x)
    return ((1 / len(y)) *  np.sum(abs_d((y_predict) - y) * model.derrivative_w(x)), # dw
            (1 / len(y)) *  np.sum(abs_d((y_predict) - y) * model.derrivative_b(x))) # db

def relu(x):
    return max(0, x)

def relu_d(x):
    x = x.copy()
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

class LinearModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def grad_update(self, w, b):
        self.w -= np.array([[w]])
        self.b -= np.array([b]) 

    def forward(self, x):
        return x * w[:, [0]] + b
    
    def derrivative_w(self, x):
        return x

    def derrivative_b(self, x):
        return 1

class Model:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def grad_update(self, w, b):
        self.w -= np.array([[w]])
        self.b -= np.array([b]) 

    def forward(self, x):
        return max(0, x * w[:, [0]] + b)
    
    def derrivative_w(self, x):
        return x

    def derrivative_b(self, x):
        return 1


# %%


a = 1.0
b_int = 0.0
w = np.array([[a]])
b = np.array([b_int])
model = Model(w, b)
plot = plot_model(x, y, model)
prev_ad = 0
u = 0.3

maes = []
iterations = 1000
for i in range(iterations):
    grad_w, grad_b = mae_d(y, x, model.w, model.b, model)
    model.grad_update(u * grad_w, u * grad_b)
    maes.append(mae(y, x, model.w, model.b))

print(f'a: {a} b: {b}, MAE: {min(maes)}')
plot_model(x, y, model, plot=plot)


# %%
plt.plot(np.arange(iterations), maes)
print(min(maes))


# %%
