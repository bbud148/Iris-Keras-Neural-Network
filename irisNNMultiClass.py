import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

# import csv
csv = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
iris = np.genfromtxt(csv, delimiter = ',', dtype = None)

# Munge data and create binary classes
iris = np.delete(iris, 0, 0)  # delete header row

# split into data and label classes

train = iris[np.r_[0:40, 50:90, 100:140], 0:4]
train = train.astype(float)

test = iris[np.r_[40:50, 90:100, 140:150], 0:4]
test = test.astype(float)

labels = np.arange(0, 3)
labels = np.repeat(labels, 40)
labels = to_categorical(labels, 3)

# develop NN model
model = Sequential()
model.add(Dense(3, input_dim = 4))
model.add(Activation('softmax'))
sgd = SGD(lr=0.1, decay=.1, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# fit NN
model.fit(train, labels, nb_epoch = 90, batch_size = 32)

# evaluate NN 
scores = model.evaluate(train, labels)

# predict
prediction = model.predict(test, batch_size = 10)
print '\n setosa\n', prediction[1:10]
print '\n versicolor\n', prediction[10:20]
print '\n virginica\n', prediction[20:30]