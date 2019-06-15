# This is a very simple fashion classifier
# It uses a data set called "fashion mnist", a set of small b&w images of apparel
# 
# This training script classifies with around 87% accuracy currently.
#
# Can you get the validation accuracy (val_acc) above 90% with a CNN?
#
# The most common error looks something like:
# ValueError: Error when checking input: expected conv2d_input to have 4 dimensions, 
#    but got array with shape (60000, 28, 28)
#
# This means you need to reshape your b&w image input from (28, 28) to (28, 28, 1)
# You can use numpy.reshape or the keras Reshape layer.
# There is example code in the examples/keras-cnn directory.
#
# Getting the CNN to work better than a multi-layer perceptron on this dataset
# may take some experimentation. 


from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape, Dropout,LeakyReLU,ELU
import tensorflow
import keras
import wandb
from wandb.keras import WandbCallback

def swish(x):
   beta = 1
   return beta * x * keras.backend.sigmoid(x)
# logging code
run = wandb.init(project="fashion")
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3
config.first_layer_conv_height = 3
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 28
config.img_height = 28
config.epochs = 10
config.activation = swish
config.activation_layers="swish"
#config.activation_layers = keras.layers.ELU(alpha=1.0)
#config.activation_layers = PReLU(alpha_initializer='zeros', weights=None)
#
# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]
X_train = X_train.reshape(
    X_train.shape[0], 28,28, 1)
X_test = X_test.reshape(
    X_test.shape[0], 28, 28, 1)
# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress",
          "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


X_train = X_train / 255.
X_test = X_test / 255.

num_classes = y_train.shape[1]

# create model
model = Sequential()
model.add(Conv2D(32,
                 (config.first_layer_conv_width, config.first_layer_conv_height),
                 input_shape=(28, 28, 1),
                 activation=config.activation))
#model.add(ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,
                 (3, 3),
                 activation=config.activation))
#model.add(ELU(alpha=1.0))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(config.dense_layer_size, activation=config.activation))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Fit the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels)])
