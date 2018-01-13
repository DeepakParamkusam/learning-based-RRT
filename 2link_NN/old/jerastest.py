from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error

# data = np.loadtxt('../training_data/raw/control_data_100k.txt')
data = np.loadtxt('../training_data/raw/control_3k_lagrange.txt')


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

model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(40,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

#training
model.fit(x_t, y_t, epochs=100, batch_size=250)

#validation
scores = model.evaluate(x_v, y_v)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))