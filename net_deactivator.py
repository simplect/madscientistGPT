# %%
import numpy as np
import pandas as pd
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
#%%
def plot_model(x, y, model, plot=None):
    if plot:
        fig, ax = plot
    else:
        fig, ax = plt.subplots()

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
   # ax.set_xlim((-10, 755000.0))
    ax.set_ylim((-10, 755000.0)) 
    #ax.scatter(x1, y, label='data')
    ax.scatter(model.forward(x), y, label='prediction', marker='+')
    
    #ax.axline((0, model.b[0]), slope=model.w[0,0], color='C2', label='model')
    ax.axline((0, 0), slope=1, color='C2', label='model')
    fig.legend()
    fig.show()
    return (fig, ax)
#plot = plot_model(x, y, model)

#%%
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
    return np.where(x > 0, 1, 0.001)

def sigmoid(x):
    return np.log(1 + np.exp(x))

def sigmoid_d(x):
    return np.exp(x) / (1 + np.exp(x))



class Model:
    def __init__(self, input_size, hidden_size):
        self.w = np.random.normal(4,size=hidden_size * input_size).reshape(input_size,-1)
        self.b = np.zeros((hidden_size))
        self.w_output = np.random.normal(4, size=hidden_size).reshape(-1,1)
        self.linear_cache = None
        self.activation = sigmoid
        self.activation_d = sigmoid_d
        self.activation = relu
        self.activation_d = relu_d

    def grad_update(self, w, b, w_output):
        self.w -= w 
        self.b -= np.array(b) 
        self.w_output -= w_output.reshape(-1,1) 
        self.linear_cache = None
    
    def linear_pass(self, x):
        #if self.linear_cache is None:
        self.linear_cache = x @ self.w + self.b
        return self.linear_cache
    
    def forward(self, x):
        # cache this?
        return self.activation(self.linear_pass(x)) @ self.w_output

    def derrivative_w_output(self, x):
        return self.activation(self.linear_pass(x))

    def derrivative_w(self, x, i, j):
        return self.activation_d(self.linear_pass(x))[0,j] * x[0,i]

    def derrivative_b(self, x, i):
        return self.activation_d(self.linear_pass(x))[0,i]
    
  

def mae_d(x, y, model):
    y_predict = model.forward(x)
    return (np.array([np.sum(abs_d((y_predict) - y) 
                                   * model.derrivative_w(x, i, j))
                      for i in range(model.w.shape[0])
                      for j in range(model.w.shape[1])]
                        ).reshape(model.w.shape), # dw
            np.array([np.sum(abs_d((y_predict) - y) 
                                   * model.derrivative_b(x,i))
                      for i in range(model.b.shape[0])]),
            np.sum(abs_d((y_predict) - y) 
                      * model.derrivative_w_output(x), axis=0)) # db

#%% 
df = (pd.read_csv('train.csv').select_dtypes(include='number'))
xy = (df
        #.replace('#', '-1.0')
        .fillna(0)
        .astype(np.float32)
        .to_numpy()
        )
np.random.shuffle(xy)
split_idx = round(xy.shape[0] * 0.7)
train = xy[:split_idx,:]
test = xy[split_idx:]


hidden_size = 30
model = Model(xy.shape[1]-1, hidden_size)
#plot = plot_model(x, y, model)

u = 0.0000001

maes = []
iterations = 1000
for iter in range(iterations):
    xys = train.copy()
    np.random.shuffle(xys)
    for i in range(xys.shape[0]):
        x = xys[[i], :-1]
        y = xys[[i], [-1]]
        
        grad_w, grad_b, grad_w_ouput = mae_d(x, y, model)
        mae_s = mae(x, y, model)
        mae_term = u * mae_s * 0
        model.grad_update(u  * grad_w,
                          u  * grad_b,
                          u  * grad_w_ouput)
    
    x = xys[:, :-1]
    y = xys[:, [-1]]
    mae_it = mae(x, y, model)
    print(f'Iteration {iter} finished with MAE: {mae_it}')
    maes.append(mae_it)


print(f'W: {model.w} \nb: {model.b}, \nMAE: {min(maes)}')
print(f'W-output: {model.w_output}')
plot_model(x, y, model)


2# %%
plt.scatter(np.arange(len(maes)), maes)
print(min(maes))


# %%
