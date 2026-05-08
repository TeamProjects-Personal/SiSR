import numpy as np
import csv 
import random
import math
#Neural Network for Recognizing color of a flower depending upon the width of petals and the number of petals

# Input will be Width X1 and Number of Petals X2 Matric of (2x1)
# Weights W1 will be from each input to each nueron in Hidden layer ie Matrix of (3x2) 
# Hidden layer of 3 neuron Matrix of (3x1)
# Weights W1 will be from each Hidden layer neuron to each output ie Matrix of (2x3) 
# Ouput will either red or purple flower so Matrix of (2x1)


def GenerateDataSet():
    with open("DataSet_Flowers.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["petal_width", "petal_count", "label"])
     # Red flowers
        for _ in range(100):
            width = round(random.uniform(1.2, 2.0), 2)
            petals = random.randint(5, 8)
            writer.writerow([width, petals, "Red"])

        # Purple flowers
        for _ in range(100):
            width = round(random.uniform(2.3, 3.2), 2)
            petals = random.randint(9, 12)
            writer.writerow([width, petals, "Purple"])
#GenerateDataSet()

dataset = open("DataSet_Flowers.csv","r")
content = dataset.read().split("\n")

Weights1 = []
Weights2 = []
def GenerateWeights(Weights):
    for _ in range(6):
        Weights.append(round(random.uniform(-0.5,0.5),2))
GenerateWeights(Weights1)
GenerateWeights(Weights2)

bias1=[]
bias2=[]
def GenerateBiases(bias,n):
    for _ in range(n):
        bias.append(round(random.uniform(-1,1),2))
GenerateBiases(bias1,3)
GenerateBiases(bias2,2)




def Sigmoid(Input):
    return 1 / (1 + np.exp(-Input))

InnerContent = content[1].split(",")



X = [float(InnerContent[0]),float(InnerContent[1])]

W1 = np.array(Weights1).reshape(3,2)
B1 = np.array(bias1).reshape(3,1)

Multiplied_weights_W1 = np.array(W1 @ X).reshape(3,1) + B1 

A1 = np.array(Sigmoid(Multiplied_weights_W1)).reshape(3,1)
W2 = np.array(Weights2).reshape(2,3)
B2 = np.array(bias2).reshape(2,1)
Multiplied_weights_W2 = np.array(W2 @ A1).reshape(2,1) + B2 
out = np.array(Sigmoid(Multiplied_weights_W2)).reshape(2,1)







