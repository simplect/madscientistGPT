# %%
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
%config inlinebackend.figure_format='retina'

# %%
x = np.arange(10).reshape(-1,1)
y = ((20) 
     * x 
     + 50
    # + (np.random.randint(0, 20, size=10)
      #           .reshape(-1,1))
    )
w = np.array([[2]])
b = np.array([0])
x.shape, y.shape, w.shape

# %%
def plot_model(x, y, w, b, plot=None):
    if plot:
        fig, ax = plot
    else:
        fig, ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(x, y, label='data')
    ax.axline((0, b[0]), slope=w[0,0], color='C2', label='model')
    fig.legend()
    fig.show()
    return (fig, ax)

plot = plot_model(x, y, w, b)
plot_model(x, y+1, w, b, plot=plot)


# %%
def abs(x):
    return np.abs(x)

def mean(x):
    return 

def mae(y, x, w, b):
    return (1 / len(x)) * np.sum(abs((x * w[:, [0]] + b[0]) - y))

mae(y, x, w, b)



# %%
def abs_d(x):
    x = x.copy()
    x[x < 0] = -1
    x[x >= 0] = 1
    return x
    #return x / (np.abs(x) + 0.00001)

def mae_d(y, x, w, b):
    return ((1 / len(y)) *  np.sum(abs_d((x * w[:, [0]] + b) - y) * x ), # dw
            (1 / len(y)) *  np.sum(abs_d((x * w[:, [0]] + b) - y))) # db

mae_d(y, x, w, b)



# %%

w = np.array([[a]])
b = np.array([b_int])
plot = plot_model(x, y, w, b)
a = 1
b_int = 0
prev_ad = 0
u = 0.3

maes = []
iterations = 1000
for i in range(iterations):
    w = np.array([[a]])
    b = np.array([b_int])
    grad_w, grad_b = mae_d(y, x, w, b)

    a -= u * grad_w
    b_int -= u * grad_b
    maes.append(mae(y, x, w, b))

print(f'a: {a} b: {b}, MAE: {min(maes)}')
plot_model(x, y, w, b, plot=plot)


# %%
plt.plot(np.arange(iterations), maes)
print(min(maes))


# %%
