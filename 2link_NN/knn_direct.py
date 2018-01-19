import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

YscaledFlag = False #FALSE IF OUTPUT IS NOT SCALED
learnCost = True
input_constrained = False

if input_constrained == True:
	if learnCost == True:
		data = np.loadtxt('../training_data/clean_direct/cost_lagrange_cons_clean.txt',delimiter=',')
		k_cost = 4
	else:
		data = np.loadtxt('../training_data/clean_direct/control_lagrange_cons_clean.txt',delimiter=',')
		k_control = 22

else:
	if learnCost == True:
		data = np.loadtxt('../training_data/clean_direct/cost_jan12_bound.txt',delimiter=',')
		# data = np.loadtxt('../training_data/clean_direct/cost_jan12.txt')
		k_cost = 13
	else:
		# data = np.loadtxt('../training_data/clean_direct/control_jan12_bound.txt',delimiter=',')
		data = np.loadtxt('../training_data/clean_direct/control_data_jan12.txt')
		k_control = 18


x = data[:,0:8]
if learnCost == True:
	y_cost = data[:,8]
	y_cost = y_cost.reshape(-1,1)
else:
	y_control = data[:,8:48]

X = x
if learnCost == False:
	Y = y_control
	k = k_control
else:
	Y = y_cost
	k = k_cost

xscaler = StandardScaler()
xscaler.fit(X)
X = xscaler.transform(X)
if YscaledFlag == True:
	yscaler = StandardScaler()
	yscaler.fit(Y)
	Y = yscaler.transform(Y)

#K-FOLD CROSS VALIDATION + PREDICTION PROFILE
kf = KFold(n_splits = 10, shuffle = True, random_state = 11)
mse = []

for train_index, test_index in kf.split(X):
	# print train_index
	knn = neighbors.KNeighborsRegressor(k, weights='uniform')
	knn.fit(X[train_index],Y[train_index])
	predicted = knn.predict(X[test_index])
	mse_fold = mean_squared_error(Y[test_index],predicted)
	mse.append(mse_fold)
	if mse_fold == min(mse):
		final_index = [train_index,test_index]
	
print min(mse)

knn = neighbors.KNeighborsRegressor(k, weights='uniform')
knn.fit(X[final_index[0]],Y[final_index[0]])
pred = knn.predict(X[final_index[1]])

fig, ax = plt.subplots()
ax.scatter(Y[final_index[1]], pred, marker=".", edgecolors=(0, 0, 0))
ax.plot([Y[final_index[1]].min(), Y[final_index[1]].max()], [Y[final_index[1]].min(), Y[final_index[1]].max()], 'k--', lw=4)
if learnCost == False:
	ax.set_xlabel('Actual control')
	ax.set_ylabel('Predicted control')
else:
	ax.set_xlabel('Actual cost')
	ax.set_ylabel('Predicted cost')
plt.show()

joblib.dump(knn, '../trained_models/knn-direct-uncons-cost.pkl')




