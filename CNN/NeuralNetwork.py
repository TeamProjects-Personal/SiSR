import numpy as np
import random
import math

class NeuralNetwork1:
    def __init__(self,layer_sizes):
    
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.weights = [np.round(np.random.standard_normal(s)/s[1]**0.5,2) for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]] 

    def predict(self,a):
        for w,b in zip(self.weights,self.biases):
            a = self.activation((w @ a + b))
        return a

    def activation(self,x):
        return 1 / (1 + np.exp(-x)) 

    def print_accuracy(self, images,lables):
        prediction = self.predict(images)
        correct = sum([np.argmax(a) ==  np.argmax(b) for a,b in zip(prediction,lables)])
        print("{0}/{1} : acc = {2}".format(correct,len(images),(correct/len(images)*100)))

    def cost(self,a,y):
        diff = (a - y)**2 # for all a & y (j) elements
        r_cost = sum(diff);
        return r_cost
        
    def backprop(self,x,b,w1):   
        z = w1 @ x
        
                
        
        


