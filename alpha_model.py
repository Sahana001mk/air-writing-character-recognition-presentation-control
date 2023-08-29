import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pickle


data = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
X = data.drop('0', axis=1)
y = data['0']

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# reshaping
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

# shuffle training data
shuffle_data = shuffle(x_train)

# visualize
'''fig, axes = plt.subplots(3,3, figsize=(10,10))
axes = axes.flatten()
for i in range(9):
    _, shu = cv2.threshold(shuffle_data[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuffle_data[i], (28,28)),cmap="Greys")
plt.show()'''

# again reshaping
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# converting to to_categorical
y_training = to_categorical(y_train, num_classes=26, dtype='int')
y_testing = to_categorical(y_test, num_classes=26, dtype='int')

# model creation
model = Sequential()
# 3 convolutional layers (Conv2D) of 64,64,64 layers each
# MaxPool layers to reduce the number of features extracted
# Flatten the layers, then two fully connected layers (Dense) of 128265 layers respectively
# output layer with softmax as activation function
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(26, activation="softmax"))

# print(model.summary())

# compile and fit model
model.compile(optimizer= Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
history = model.fit(x_train, y_training, epochs=5, validation_data=(x_test, y_testing))

with open('alphaCnn.pickle', 'wb') as f:
    pickle.dump(model, f)

# save model
# model.save(r'handwritten_character_recog_model.h5')

# create words dictionary


'''fig, axes = plt.subplots(3,3, figsize=(8,9))
axes = axes.flatten()
for i,ax in enumerate(axes):
    image = np.reshape(x_test[i], (28,28))
    ax.imshow(image, cmap="Greys")
    pred = words[np.argmax(y_testing[i])]
    ax.set_title("Prediction: "+pred)
    ax.grid()
plt.show()'''