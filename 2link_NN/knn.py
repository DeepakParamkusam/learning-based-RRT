import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#INDIRECT CONS
# x = np.loadtxt('../training_data/clean_indirect/X100kc.txt',delimiter=',')
# y = np.loadtxt('../training_data/clean_indirect/Y100kc.txt',delimiter=',')

#INDIRECT UNCONS
# x = np.loadtxt('../training_data/clean_indirect/X100k.txt',delimiter=',')
# y = np.loadtxt('../training_data/clean_indirect/Y100k.txt',delimiter=',')

#FOR INDIRECT
# y_cost = y[:,0]
# y_control = y[:,1:]

#DIRECT UNCONS
# data = np.loadtxt('../training_data/clean_direct/control_100k_clean_4.txt')
# data = np.loadtxt('../training_data/clean_direct/cost_100k_clean_4.txt')

#DIRECT CONS
# data = np.loadtxt('../training_data/raw/control_data_100k.txt')
# data = np.loadtxt('../training_data/corrected_clean_data/2_costr_100k_clean_7.txt')

#NEW DIRECT
# data = np.loadtxt('../training_data/clean_direct/cost_jan12.txt')
data = np.loadtxt('../training_data/clean_direct/control_data_jan12.txt')

#FOR DIRECT
x = data[:,0:8]
# y_cost = data[:,8]
y_control = data[:,8:]

#SCALING
# y_cost = y_cost.reshape(-1,1)
YscaledFlag = True #false if y is not to be scaled
x_t, x_v, y_t, y_v = train_test_split(x, y_control, test_size=0.1, random_state = 42)

xscaler = StandardScaler()
yscaler = StandardScaler()

xscaler.fit(x_t)
yscaler.fit(y_t)

x_t = xscaler.transform(x_t)
x_v = xscaler.transform(x_v)
if YscaledFlag == True:
	y_t = yscaler.transform(y_t)
	y_v = yscaler.transform(y_v)

#MSE AND PREDICTION PROFILE k=11 (control) k=8(cost) 
knn = neighbors.KNeighborsRegressor(11, weights='uniform')
predicted = cross_val_predict(knn, x_t, y_t,cv=10)
scores = cross_val_score(knn, x_t, y_t,cv=10,scoring='neg_mean_squared_error')
print -scores.mean() 

fig, ax = plt.subplots()
ax.scatter(y_t, predicted, edgecolors=(0, 0, 0))
ax.plot([y_t.min(), y_t.max()], [y_t.min(), y_t.max()], 'k--', lw=4)
ax.set_xlabel('Actual cost')
ax.set_ylabel('Predicted cost')
plt.show()

#VALIDATION
# knn.fit(x_t,y_t)
# pred=knn.predict(x_v)
# fig, ax = plt.subplots()
# ax.scatter(y_v, pred, edgecolors=(0, 0, 0))
# ax.plot([y_v.min(), y_v.max()], [y_v.min(), y_v.max()], 'k--', lw=4)
# ax.set_xlabel('Actual cost')
# ax.set_ylabel('Predicted cost')
# plt.show()

#BEST K
# neigh = np.arange(2,25)
# a=[]
# for k in neigh:
# 	knn = neighbors.KNeighborsRegressor(k, weights='uniform')
# 	score = cross_val_score(knn, x_t, y_t,cv=10,scoring='neg_mean_squared_error')
# 	a.append(-score.mean())

# print a
# plt.plot(neigh,a)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Error')
# plt.show()


