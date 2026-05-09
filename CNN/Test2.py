import ConvolutionNeuralNetwork as NN;
import numpy as np;
import cv2 as cv
import glob



hr_path = "./Dataset/train/HR/*.jpg"
lr_path = "./Dataset/train/LR/*.jpg"

HR = []
LR = []

hr_files = sorted(glob.glob(hr_path))
lr_files = sorted(glob.glob(lr_path))

for hr_file, lr_file in zip(hr_files, lr_files):

    hr_img = cv.imread(hr_file)
    lr_img = cv.imread(lr_file)

    if hr_img is None or lr_img is None:
        continue

    HR.append(hr_img)
    LR.append(lr_img)

HR = np.array(HR, dtype=object)
LR = np.array(LR, dtype=object)



net = NN.ConvolutionNeuralNetwork(16,learning_rate=0.1)




LR_up = []

for lr, hr in zip(LR, HR):

    h, w = hr.shape[:2]

    lr_up = cv.resize(lr, (w, h), interpolation=cv.INTER_CUBIC)

    LR_up.append(lr_up)

LR_up = np.array(LR_up, dtype=object)


net.train(LR_up,HR,10)


