import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = np.loadtxt('../training_data/raw/control_3k_lagrange.txt')
# data = np.loadtxt('../training_data/corrected_clean_data/2_control_data_fulltraj_100k_clean_5.txt')
# data = np.loadtxt('../training_data/raw/2_control_data_unconstrained.txt')
# data = np.loadtxt('../training_data/raw/control_data_100k.txt')

x = data[:,0:8]
y = data[:,8:48]

num_data = len(x)
x_t = x[0:int(0.9*num_data),:]
x_v = x[int(0.9*num_data):num_data,:]
y_t = y[0:int(0.9*num_data),:]
y_v = y[int(0.9*num_data):num_data,:]

# print(np.min(y_t,axis=0))
scaler = StandardScaler()
b=StandardScaler()

scaler.fit(x_t)
b.fit(y_t)

x_t = scaler.transform(x_t)
x_v = scaler.transform(x_v)
y_t=b.transform(y_t)
y_v=b.transform(y_v)

mlp = MLPRegressor(hidden_layer_sizes=(8),max_iter=1000)
mlp.fit(x_t,y_t)
pred = mlp.predict(x_v)
mse = explained_variance_score(b.inverse_transform(y_v),b.inverse_transform(pred))
print "mse =", mse

# y_p = mlp.predict(scaler.transform(x[0:1,0:8]))
# print("pred=",b.inverse_transform(y_p))
# print("actual=",y[0:1,8:48])
# # plot_knn(y_v,pred,flag)
q=50
t = np.arange(0.0,40.0,1.0)
plt.figure(1)
plt.plot(t,b.inverse_transform(y_v[q,:]))
plt.plot(t, b.inverse_transform(pred[q,:]))
# plt.plot(t,y_v[q,:])
# plt.plot(t, pred[q,:])
plt.xlabel('time(x0.025)')
plt.ylabel('scaled torque')
plt.show()