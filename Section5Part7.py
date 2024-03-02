#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: su
"""

import numpy as np, array
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
from keras.callbacks import Callback


#5
Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []
Z1 = []
Z2 = []
Z3 = []
Z4 = []
Z5 = []

for i in range(1,6):
    temp = pd.read_csv('Y' + str(i) + '.csv', header = 0)
    exec(f'Y{i} = temp')
  
for i in range(1,6):
    temp = pd.read_csv('Z' + str(i) + '.csv', header = 0)
    exec(f'Z{i} = temp')
    
# Y1 = Y1.reset_index(drop=True,inplace=True)
  

#change stopping rule
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_loss = logs["loss"]
        if val_loss <= self.threshold:
            self.model.stop_training = True



# Build network

# N1
def model1_create():
    model = Sequential()
    model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation="tanh"))
    model.add(Dense(1, activation = "sigmoid"))
    model.compile(loss="mse", optimizer="Adam", metrics=['mse'])
    return model

model1 = model1_create()
model1.summary()

# N2
def model2_create():
    model = Sequential()
    model.add(Dense(4, input_dim=1, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="mse", optimizer="Adam", metrics=['mse'])
    return model

model2 = model2_create()
model2.summary()



# N3
def model3_create():
    model = Sequential()
    model.add(Dense(2, input_dim=1, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="mse", optimizer="Adam", metrics=['mse'])
    return model

model3 = model3_create()
model3.summary()




# N4
def model4_create():
    model = Sequential()
    model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="mse", optimizer="Adam", metrics=['mse'])
    return model

model4 = model4_create()
model4.summary()

epoch = 10000


#E1
#a
my_callback = MyThresholdCallback(threshold=0.065) 
X_train = Y1[Y1.columns[0]].values
Y_train = Y1[Y1.columns[1]].values
X_val = Y5[Y5.columns[0]].values
Y_val = Y5[Y5.columns[1]].values

model1 = model1_create()
resultE1_a_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=epoch,
                           callbacks=[my_callback], batch_size=1, verbose=0)


results = resultE1_a_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_a_N1.history['loss'])))



model2 = model2_create()
resultE1_a_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_a_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_a_N2.history['loss'])))



model3 = model3_create()
resultE1_a_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_a_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_a_N3.history['loss'])))




model4 = model4_create()
resultE1_a_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_a_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_a_N4.history['loss'])))



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E1a validation MSE from Network N1 - N4')
ax1.plot(resultE1_a_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE1_a_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE1_a_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE1_a_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    
    
    
#b
my_callback = MyThresholdCallback(threshold=0.07)  

X_train = Z1[Z1.columns[0]].values
Y_train = Z1[Z1.columns[1]].values
X_val = Z5[Z5.columns[0]].values
Y_val = Z5[Z5.columns[1]].values

model1 = model1_create()
resultE1_b_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback],  batch_size=1, verbose=0)

results = resultE1_b_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_b_N1.history['loss'])))


model2 = model2_create()
resultE1_b_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_b_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_b_N2.history['loss'])))



model3 = model3_create()
resultE1_b_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_b_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_b_N3.history['loss'])))




model4 = model4_create()
resultE1_b_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_b_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE1_b_N4.history['loss'])))



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E1b validation MSE from Network N1 - N4')
ax1.plot(resultE1_b_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE1_b_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE1_b_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE1_b_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    


#E2
#a
my_callback = MyThresholdCallback(threshold=0.025)  

X_train = Y1[Y1.columns[0]].values
X_train = np.append(X_train,Y2[Y2.columns[0]].values)
Y_train = Y1[Y1.columns[1]].values
Y_train = np.append(Y_train, Y2[Y2.columns[1]].values)
X_val = Y5[Y5.columns[0]].values
Y_val = Y5[Y5.columns[1]].values

model1 = model1_create()
resultE2_a_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_a_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_a_N1.history['loss'])))


model2 = model2_create()
resultE2_a_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE1_a_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_a_N2.history['loss'])))


model3 = model3_create()
resultE2_a_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_a_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_a_N3.history['loss'])))



model4 = model4_create()
resultE2_a_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_a_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_a_N4.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E2a validation MSE from Network N1 - N4')
ax1.plot(resultE2_a_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE2_a_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE2_a_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE2_a_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    
    
    
#b
my_callback = MyThresholdCallback(threshold=0.07)  
X_train = Z1[Z1.columns[0]].values
X_train = np.append(X_train,Z2[Z2.columns[0]].values)
Y_train = Z1[Z1.columns[1]].values
Y_train = np.append(Y_train, Z2[Z2.columns[1]].values)
X_val = Z5[Z5.columns[0]].values
Y_val = Z5[Z5.columns[1]].values


model1 = model1_create()
resultE2_b_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_b_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_b_N1.history['loss'])))

model2 = model2_create()
resultE2_b_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_b_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_b_N2.history['loss'])))


model3 = model3_create()
resultE2_b_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_b_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_b_N3.history['loss'])))



model4 = model4_create()
resultE2_b_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE2_b_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE2_b_N4.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E2b validation MSE from Network N1 - N4')
ax1.plot(resultE2_b_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE2_b_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE2_b_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE2_b_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    






#E3
my_callback = MyThresholdCallback(threshold=0.025)  

#a

X_train = Y1[Y1.columns[0]].values
X_train = np.append(X_train,Y2[Y2.columns[0]].values)
X_train = np.append(X_train,Y3[Y3.columns[0]].values)
Y_train = Y1[Y1.columns[1]].values
Y_train = np.append(Y_train, Y2[Y2.columns[1]].values)
Y_train = np.append(Y_train, Y3[Y3.columns[1]].values)
X_val = Y5[Y5.columns[0]].values
Y_val = Y5[Y5.columns[1]].values

model1 = model1_create()
resultE3_a_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_a_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_a_N1.history['loss'])))

model2 = model2_create()
resultE3_a_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_a_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_a_N2.history['loss'])))


model3 = model3_create()
resultE3_a_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_a_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_a_N3.history['loss'])))




model4 = model4_create()
resultE3_a_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_a_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_a_N1.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E3a validation MSE from Network N1 - N4')
ax1.plot(resultE3_a_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE3_a_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE3_a_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE3_a_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    
    
    
#b
my_callback = MyThresholdCallback(threshold=0.07)  
X_train = Z1[Z1.columns[0]].values
X_train = np.append(X_train,Z2[Z2.columns[0]].values)
X_train = np.append(X_train,Z3[Z3.columns[0]].values)
Y_train = Z1[Z1.columns[1]].values
Y_train = np.append(Y_train, Z2[Z2.columns[1]].values)
Y_train = np.append(Y_train, Z3[Z3.columns[1]].values)
X_val = Z5[Z5.columns[0]].values
Y_val = Z5[Z5.columns[1]].values


model1 = model1_create()
resultE3_b_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_b_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_b_N1.history['loss'])))

model2 = model2_create()
resultE3_b_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_b_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_b_N2.history['loss'])))


model3 = model3_create()
resultE3_b_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_b_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_b_N3.history['loss'])))



model4 = model4_create()
resultE3_b_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE3_b_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE3_b_N4.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E3b validation MSE from Network N1 - N4')
ax1.plot(resultE3_b_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE3_b_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE3_b_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE3_b_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    

#E4
my_callback = MyThresholdCallback(threshold=0.025)  

#a

X_train = Y1[Y1.columns[0]].values
X_train = np.append(X_train,Y2[Y2.columns[0]].values)
X_train = np.append(X_train,Y3[Y3.columns[0]].values)
X_train = np.append(X_train,Y4[Y4.columns[0]].values)
Y_train = Y1[Y1.columns[1]].values
Y_train = np.append(Y_train, Y2[Y2.columns[1]].values)
Y_train = np.append(Y_train, Y3[Y3.columns[1]].values)
Y_train = np.append(Y_train, Y4[Y4.columns[1]].values)
X_val = Y5[Y5.columns[0]].values
Y_val = Y5[Y5.columns[1]].values

model1 = model1_create()
resultE4_a_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_a_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_a_N1.history['loss'])))



model2 = model2_create()
resultE4_a_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_a_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_a_N2.history['loss'])))


model3 = model3_create()
resultE4_a_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_a_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_a_N3.history['loss'])))



model4 = model4_create()
resultE4_a_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_a_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_a_N4.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E4a validation MSE from Network N1 - N4')
ax1.plot(resultE4_a_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE4_a_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE4_a_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE4_a_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    
    
    
#b
my_callback = MyThresholdCallback(threshold=0.07)  
X_train = Z1[Z1.columns[0]].values
X_train = np.append(X_train,Z2[Z2.columns[0]].values)
X_train = np.append(X_train,Z3[Z3.columns[0]].values)
X_train = np.append(X_train,Z4[Z4.columns[0]].values)
Y_train = Z1[Z1.columns[1]].values
Y_train = np.append(Y_train, Z2[Z2.columns[1]].values)
Y_train = np.append(Y_train, Z3[Z3.columns[1]].values)
Y_train = np.append(Y_train, Z4[Z4.columns[1]].values)
X_val = Z5[Z5.columns[0]].values
Y_val = Z5[Z5.columns[1]].values


model1 = model1_create()
resultE4_b_N1 = model1.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_b_N1
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_b_N1.history['loss'])))

model2 = model2_create()
resultE4_b_N2 = model2.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_b_N2
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_b_N2.history['loss'])))


model3 = model3_create()
resultE4_b_N3 = model3.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_b_N3
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_b_N3.history['loss'])))



model4 = model4_create()
resultE4_b_N4 = model4.fit(X_train, Y_train, validation_data = (X_val, Y_val), 
                          epochs=epoch, callbacks=[my_callback], batch_size=1, verbose=0)

results = resultE4_b_N4
max_loss = np.max(results.history['val_loss'])
min_loss = np.min(results.history['val_loss'])
mean_loss = np.mean(results.history['val_loss'])
print("Maximum MSE : {:.4f}".format(max_loss))
print("")
print("Minimum MSE : {:.4f}".format(min_loss))
print("")
print("Mean MSE : {:.4f}".format((mean_loss)))
print("")
print("Number of epochs to reach the threshold is " + str(len(resultE4_b_N4.history['loss'])))


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
plt.suptitle('E4b validation MSE from Network N1 - N4')
ax1.plot(resultE4_b_N1.history['val_loss'])
ax1.set_title('N1')
ax1.set(xlabel='epoch', ylabel='mse')
ax2.plot(resultE4_b_N2.history['val_loss'])
ax2.set_title('N2')
ax2.set(xlabel='epoch', ylabel='mse')
ax3.plot(resultE4_b_N3.history['val_loss'])
ax3.set_title('N3')
ax3.set(xlabel='epoch', ylabel='mse')
ax4.plot(resultE4_b_N4.history['val_loss'])
ax4.set_title('N4')
ax4.set(xlabel='epoch', ylabel='mse')

for ax in fig.get_axes():
    ax.label_outer()
    

