import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

YscaledFlag = False #FALSE IF OUTPUT IS NOT SCALED
learnCost = True
input_constrained = True

if input_constrained == True:
	#INDIRECT CONS DATA
	x = np.loadtxt('../training_data/clean_indirect/X100kc.txt',delimiter=',')
	y = np.loadtxt('../training_data/clean_indirect/Y100kc.txt',delimiter=',')
else:
	#INDIRECT UNCONS DATA
	x = np.loadtxt('../training_data/clean_indirect/X100k.txt',delimiter=',')
	y = np.loadtxt('../training_data/clean_indirect/Y100k.txt',delimiter=',')

y_cost = y[:,0]
y_control = y[:,1:]
y_cost = y_cost.reshape(-1,1)

X = x
if learnCost == False:
	Y = y_control
else:
	Y = y_cost

xscaler = StandardScaler()
xscaler.fit(X)
X = xscaler.transform(X)

if YscaledFlag == True:
	yscaler = StandardScaler()
	yscaler.fit(Y)
	Y = yscaler.transform(Y)

#K-FOLD CROSS VALIDATION + PREDICTION PROFILE
# kf = KFold(n_splits = 10, shuffle = True, random_state = 11)
# mse = []

# for train_index, test_index in kf.split(X):
# 	# print train_index
# 	mlp = MLPRegressor(hidden_layer_sizes=(100,100),max_iter=1000,random_state=11)
# 	mlp.fit(X[train_index],Y[train_index])
# 	predicted = mlp.predict(X[test_index])
# 	mse_fold = mean_squared_error(Y[test_index],predicted)
# 	mse.append(mse_fold)
# 	if mse_fold == min(mse):
# 		final_index = [train_index,test_index]
	
# print min(mse)

# mlp = MLPRegressor(hidden_layer_sizes=(H1,H2),max_iter=1000,random_state=11)
# mlp.fit(X[final_index[0]],Y[final_index[0]])
# pred_v = mlp.predict(X[final_index[1]])

x_t, x_v, y_t, y_v = train_test_split(X, Y, test_size=0.1, random_state = 11)
mlp = MLPRegressor(hidden_layer_sizes=(100,100),random_state=11)
mlp.fit(x_t,y_t)
pred_v = mlp.predict(x_v)
mse = mean_squared_error(y_v,pred_v)
print "mse =", mse

fig, ax = plt.subplots()
ax.scatter(y_v, pred_v, marker=".", edgecolors=(0, 0, 0))
ax.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'k--', lw=4)
if learnCost == False:
	ax.set_xlabel('Actual control')
	ax.set_ylabel('Predicted control')
else:
	ax.set_xlabel('Actual cost')
	ax.set_ylabel('Predicted cost')
plt.show()

joblib.dump(mlp, '../trained_models/mlp-indirect-uncons-cost-100-100.pkl')

