
import numpy as np
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

a=[]

def plot_knn(y_v,pred,flag):
	q=01
	t = np.arange(0.0,40.0,1.0)
	plt.figure(1)
	plt.plot(t,y_v[q,:])
	plt.plot(t, pred[q,:])
	plt.xlabel('time(x0.025)')
	plt.ylabel('scaled torque')

	plt.figure(2)
	if flag==0:
		plt.plot(t,y_v[q,:]*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0))
		plt.plot(t, pred[q,:]*(y.max(axis=0)-y.min(axis=0))+y.min(axis=0))
		plt.xlabel('time(x0.025)')
		plt.ylabel('descaled torque')

		# print y.max(axis=0)-y.min(axis=0)
		# print y.min(axis=0)
	else:
		plt.scatter(t,y_v[q,:]*(y.std(axis=0))+y.mean(axis=0))
		plt.scatter(t, pred[q,:]*(y.std(axis=0))+y.mean(axis=0))
		plt.xlabel('time(x0.025)')
		plt.ylabel('descaled torque')
		# print y.mean(axis=0)
		# print y.std(axis=0) 
	plt.figure(3)
	plt.plot(t,y_v[q,:]-pred[q,:])
	plt.xlabel('time(x0.025)')
	plt.ylabel('Error')
	plt.show()

data = np.loadtxt('../training_data/raw/control_data_100k.txt')
# data = np.loadtxt('../training_data/corrected_clean_data/2_control_data_fulltraj_100k_clean_5.txt')
# data = np.loadtxt('../training_data/raw/2_control_data_unconstrained.txt')
# data = np.loadtxt('../training_data/corrected_clean_data/2_cost_100k_clean_5.txt')
# data = np.loadtxt('../training_data/corrected_clean_data/2_control_data_fulltraj_ok_clean_6.txt')
data = np.loadtxt('../training_data/raw/control_lagrange_uncons.txt')

x = data[:,0:8]
y = data[:,8:48]
# y = data[:,8:9]
# y=np.log(y)
flag = 0

if flag == 0:
	xi = np.divide(x-x.min(axis=0),x.max(axis=0)-x.min(axis=0))
	yi = np.divide(y-y.min(axis=0),y.max(axis=0)-y.min(axis=0))
else:
	xi = np.divide(x-x.mean(axis=0),x.std(axis=0))
	yi = np.divide(y-y.mean(axis=0),y.std(axis=0))

num_data = len(xi)
x_t = xi[0:int(0.9*num_data),:]
x_v = xi[int(0.9*num_data):num_data,:]
y_t = yi[0:int(0.9*num_data),:]
y_v = yi[int(0.9*num_data):num_data,:]

knn = neighbors.KNeighborsRegressor(5, weights='uniform')
pred = knn.fit(x_t, y_t).predict(x_v)
mse = mean_squared_error(y_v,pred)
print "mse =", mse
plot_knn(y_v,pred,flag)
# t=np.arange(0,len(x_v),1)
# plt.figure(1)
# plt.plot(t,y_v)
# plt.plot(t,pred)
# plt.xlabel('sample')
# plt.ylabel('cost')
# plt.show()

# neigh = np.arange(3,100,10)

# for k in neigh:
# 	knn = neighbors.KNeighborsRegressor(k, weights='uniform')
# 	score = cross_val_score(knn, xi, yi,cv=10,scoring='neg_mean_squared_error')
# 	a.append(-score.mean())

# plt.plot(neigh,a)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Error')
# plt.show()




