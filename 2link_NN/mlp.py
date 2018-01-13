import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# INDIRECT CONS
# x = np.loadtxt('../../training_data/raw_indirect/X100kc.txt',delimiter=',')
# y = np.loadtxt('../../training_data/raw_indirect/Y100kc.txt',delimiter=',')

# INDIRECT UNCONS
# x = np.loadtxt('../../training_data/raw_indirect/X100k.txt',delimiter=',')
# y = np.loadtxt('../../training_data/raw_indirect/Y100k.txt',delimiter=',')

# y_cost = y[:,0]
# y_control = y[:,1:]

# DIRECT UNCONS
# data = np.loadtxt('../../training_data/corrected_clean_data/2_control_data_fulltraj_100k_clean_4.txt')
# data = np.loadtxt('../../training_data/corrected_clean_data/2_cost_100k_clean_4.txt')
# data = np.loadtxt('../../cost_lagrange_uncons_clean2.txt',delimiter=',')


# DIRECT CONS
# data = np.loadtxt('../../training_data/corrected_clean_data/2_control_data_fulltraj_100k_clean_7.txt')
# data = np.loadtxt('../../training_data/raw/cost_3k_lagrange.txt')
# data = np.loadtxt('../../training_data/direct_constrained_clean3_cost.txt')
# data = np.loadtxt('../../cost_lagrange_uncons.txt',delimiter=',')
 
# data = np.loadtxt('../../control_lagrange_uncons.txt',delimiter=',')
#NEW DIRECT
data = np.loadtxt('../training_data/clean_direct/cost_jan12_clean.txt',delimiter=',')
# data = np.loadtxt('../training_data/clean_direct/control_jan12_clean.txt',delimiter=',')

x = data[:,0:8]
y_cost = data[:,8]
# y_control = data[:,8:48]

y_cost = y_cost.reshape(-1,1)
YscaledFlag = False

x_t, x_v, y_t, y_v = train_test_split(x, y_cost, test_size=0.1, random_state = 42)

xscaler = StandardScaler()
yscaler = StandardScaler()

xscaler.fit(x_t)
yscaler.fit(y_t)

x_t = xscaler.transform(x_t)
x_v = xscaler.transform(x_v)
if YscaledFlag == True:
	y_t = yscaler.transform(y_t)
	y_v = yscaler.transform(y_v)

# MSE AND PREDICTION PROFILE
mlp = MLPRegressor(hidden_layer_sizes=(100,100),max_iter=1000,random_state=42)
# pred = cross_val_predict(mlp, x_t, y_t,cv=10)
# scores = cross_val_score(mlp, x_t, y_t,cv=10,scoring='neg_mean_squared_error')
# print -scores.mean() 

mlp.fit(x_t,y_t)
pred = mlp.predict(x_v)
mse = mean_squared_error(y_v,pred)
print "mse =", mse

fig, ax = plt.subplots()
ax.scatter(y_v, pred, edgecolors=(0, 0, 0))
ax.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'k--', lw=4)
ax.set_xlabel('Actual cost')
ax.set_ylabel('Predicted cost')
plt.show()
