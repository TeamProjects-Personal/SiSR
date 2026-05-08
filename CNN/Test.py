import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

"""
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

plt.figure(figsize=(10, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
"""

X = X_train #whole dataset as inputs
Y = y_train #whole dataset as desired outputs
w_i_h = np.random.uniform(-0.5,0.5,(1,784)) # weight from inputs to hidden layer
h_i_o = np.random.uniform(-0.5,0.5,(1,10)) # weight from hidden layer to output layer 


#training 

def train(x,y,w1,w2,alpha,epochs):
    for epoch in range(epochs):
        for img,l in (x,y):
            
            X_in = np.array(img).shape(1,)


