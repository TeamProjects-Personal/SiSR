import ConvolutionNeuralNetwork as NN;
import numpy as np;
import cv2 as cv
import glob


hr_path = "./Dataset/train/HR/*.jpg"
lr_path = "./Dataset/train/LR/*.jpg"
HR = []
LR = []

for file in glob.glob(hr_path):
    img = cv.imread(file)
    if img is not None:
        HR.append(img)

for file in glob.glob(lr_path):
    img = cv.imread(file)
    if img is not None:
        LR.append(img)


net = NN.ConvolutionNeuralNetwork(16,learning_rate=0.1)
net.train(LR,HR,10)
#net = NN.ConvolutionNeuralNetwork(learning_rate=0.1)

#net.train()




