import NeuralNetwork2 as NN;
import numpy as np;

layer_size = [784,5,10]
with np.load("mnist.npz") as data:
    training_images = data['training_images']
    training_labels = data['training_labels']



batch_training_images = training_images
batch_training_labels = training_labels


net = NN.NeuralNetwork([784,128,64,10], learning_rate=0.1)
#net.train(batch_training_images,batch_training_labels,epochs=100)

#net.save_model("Digits_Recongnization")

net.load_model("Digits_Recongnization.npz");

index = 1000;

predection = net.predict(batch_training_images[index])

print("Prediction: " ,np.argmax(predection));
print("Actual: " ,np.argmax(batch_training_labels[index]));




