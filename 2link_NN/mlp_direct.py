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
input_constrained = False

if input_constrained == True:
	if learnCost == True:
		data = np.loadtxt('../training_data/clean_direct/cost_lagrange_cons_clean.txt',delimiter=',')		
	else:
		data = np.loadtxt('../training_data/clean_direct/control_lagrange_cons_clean.txt',delimiter=',')
else:
	if learnCost == True:
		# data = np.loadtxt('../training_data/clean_direct/cost_jan12_bound.txt',delimiter=',')
		# data = np.loadtxt('../training_data/clean_direct/cost_jan12.txt')		
		# data = np.loadtxt('../training_data/clean_direct/cost_500k_clean.txt')
		data = np.loadtxt('../training_data/clean_direct/cost_500k_bound.txt',delimiter=',')
	else:
		# data = np.loadtxt('../training_data/clean_direct/control_jan12_bound.txt',delimiter=',')
		# data = np.loadtxt('../training_data/clean_direct/control_data_jan12.txt')
		# data = np.loadtxt('../training_data/clean_direct/control_500k_clean.txt')
		data = np.loadtxt('../training_data/clean_direct/control_500k_bound.txt',delimiter=',')

		
X = data[:,0:8]
if learnCost == True:
	Y = data[:,8]
	Y = Y.reshape(-1,1)
else:
	Y = data[:,8:48]

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

joblib.dump(mlp, '../trained_models/mlp-direct-uncons-control-100-100-500k.pkl')

