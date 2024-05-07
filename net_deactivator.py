# %%
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
%config inlinebackend.figure_format='retina'

train_size = 10
x = np.arange(train_size).reshape(-1,1)
y = ((20) 
     * x ** 2
     + 50
     + (np.random.randint(0, 20, size=train_size)
                 .reshape(-1,1))
    )

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

def mae(x, y, model):
    return (1 / len(x)) * np.sum(abs(model.forward(x) - y))


def abs_d(x):
    x = x.copy()
    x[x < 0] = -1
    x[x >= 0] = 1
    return x
    #return x / (np.abs(x) + 0.00001)

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return np.log(1 + np.exp(x))

def sigmoid_d(x):
    return np.exp(x) / (1 + np.exp(x))



class Model:
    def __init__(self, hidden_size):
        self.w = np.random.normal(size=hidden_size).reshape(1,-1)
        self.b = np.zeros((hidden_size))
        self.w_output = np.random.normal(size=hidden_size).reshape(-1,1)
        self.linear_cache = None
        self.activation = sigmoid
        self.activation_d = sigmoid_d
       # self.activation = relu
       # self.activation_d = relu_d

    def grad_update(self, w, b, w_output):
        self.w -= np.array([w])
        self.b -= np.array(b) 
        self.w_output -= w_output.reshape(-1,1) 
        self.linear_cache = None
    
    def linear_pass(self, x):
        if self.linear_cache is None:
            self.linear_cache = x @ self.w + self.b
        return self.linear_cache
    
    def forward(self, x):
        # cache this?
        return self.activation(self.linear_pass(x)) @ self.w_output

    def derrivative_w_output(self, x):
        return self.activation(self.linear_pass(x))

    def derrivative_w(self, x, i):
        return self.activation_d(self.linear_pass(x))[:, [i]] * (x)

    def derrivative_b(self, x, i):
        return self.activation_d(self.linear_pass(x))
    
  

def mae_d(x, y, model):
    y_predict = model.forward(x)
    return (np.array([np.sum(abs_d((y_predict) - y) 
                                   * model.derrivative_w(x,i))
                      for i in range(model.w.shape[1])]), # dw
            np.array([np.sum(abs_d((y_predict) - y) 
                                   * model.derrivative_b(x,i))
                      for i in range(model.b.shape[0])]),
            np.sum(abs_d((y_predict) - y) 
                      * model.derrivative_w_output(x), axis=0)) # db


# %%
hidden_size = 10
model = Model(hidden_size)
plot = plot_model(x, y, model)

u = 0.0001

maes = []
iterations = 50000
for i in range(iterations):
    grad_w, grad_b, grad_w_ouput = mae_d(x, y, model)
    model.grad_update(u * grad_w,
                    u * grad_b,
                    u * grad_w_ouput)
    maes.append(mae(x, y, model))


print(f'W: {model.w} \nb: {model.b}, \nMAE: {min(maes)}')
print(f'W-output: {model.w_output}')
plot_model(x, y, model, plot=plot)


# %%
plt.plot(np.arange(iterations), maes)
print(min(maes))


# %%
