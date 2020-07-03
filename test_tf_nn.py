import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data=keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels)=data.load_data()

train_images=train_images/255.0
test_images=test_images/255.0
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model=keras.Sequential([
                        keras.layers.Flatten(),
                        keras.layers.Dense(600,activation="relu"),
                        keras.layers.Dense(128,activation="relu"),
                        keras.layers.Dense(512,activation="relu"),
                        keras.layers.Dense(10,activation="softmax")
])

model.compile(optimizer="adam",loss=loss_fn,metrics=["accuracy"])

model.fit(train_images,train_labels,epochs=10)

test_loss,test_acc=model.evaluate(test_images,test_labels)

print(test_acc)

# 98.11% accuracy
