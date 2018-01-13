import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt

x = np.loadtxt('Xv.txt',delimiter=',')
y = np.loadtxt('Yv.txt',delimiter=',')

y_cost = y[:,0]
y_cost=y_cost.reshape(-1,1)
y_control = y[:,1:]

x_t, x_v, y_t, y_v = train_test_split(x, y_cost, test_size=0.1, random_state = 1)

# xscaler = StandardScaler()
# yscaler = StandardScaler()

# xscaler.fit(x_t)
# yscaler.fit(y_t)

# x_t = xscaler.transform(x_t)
# x_v = xscaler.transform(x_v)
# y_t = yscaler.transform(y_t)
# y_v = yscaler.transform(y_v)
x_t = np.divide(x_t-x.min(axis=0),x.max(axis=0)-x.min(axis=0))
y_t = np.divide(y_t-y.min(axis=0),y.max(axis=0)-y.min(axis=0))
x_v = np.divide(x_v-x.min(axis=0),x.max(axis=0)-x.min(axis=0))
y_v = np.divide(y_v-y.min(axis=0),y.max(axis=0)-y.min(axis=0))

mlp = MLPRegressor(hidden_layer_sizes=(100),max_iter=10000,random_state=1)
predicted = cross_val_predict(mlp, x_t, y_t,cv=10)
scores = cross_val_score(mlp, x_t, y_t,cv=10,scoring='neg_mean_squared_error')
print scores

# test = mlp.fit(x_t,y_t).score(x_v,y_v)
# print test


fig, ax = plt.subplots()
ax.scatter(y_t, predicted, edgecolors=(0, 0, 0))
ax.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()