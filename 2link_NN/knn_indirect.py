import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

YscaledFlag = False #FALSE IF OUTPUT IS NOT SCALED
learnCost = False
input_constrained = True 

if input_constrained == True:
	#INDIRECT CONS DATA
	x = np.loadtxt('../training_data/clean_indirect/X100kc.txt',delimiter=',')
	y = np.loadtxt('../training_data/clean_indirect/Y100kc.txt',delimiter=',')
	k_control = 6
	k_cost = 3
else:
	#INDIRECT UNCONS DATA
	x = np.loadtxt('../training_data/clean_indirect/X100k.txt',delimiter=',')
	y = np.loadtxt('../training_data/clean_indirect/Y100k.txt',delimiter=',')
	k_control = 7
	k_cost = 2

y_cost = y[:,0]
y_control = y[:,1:]
y_cost = y_cost.reshape(-1,1)

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
yscaler = StandardScaler()
yscaler.fit(Y)
if YscaledFlag == True:
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

joblib.dump(knn, '../trained_models/knn-indirect-cons-control.pkl')




