import numpy as np
import cv2 as cv

def ExtractRegion(Image,r,c,kh,kw):
    return Image[r:r+kh, c:c+kw,:]

def ReLU(Image):
    return np.maximum(0,Image)

def Convolution(Image,kernel):

    h,w,channels = Image.shape
    kh,kw,kc = kernel.shape

    if(kc != channels):
       raise ValueError("Kernel channels must match image channels")

    out_h = h - kh + 1
    out_w = w - kw + 1

    inter_feature_map = np.zeros((out_h, out_w))
 
    for r in range(out_h):
       for c in range(out_w):

           region =np.array(ExtractRegion(Image,r,c,kh,kw))
           inter_feature_map[r,c] = np.sum(region * kernel)
    
    return np.array(inter_feature_map)

def ConvoLayer(Input,Kernels):
    num_filters = Kernels.shape[0]
    
    first_fm  = Convolution(Input,Kernels[0])
    
    h,w = first_fm.shape
    
    output = np.zeros((h,w,num_filters))

    for i in range(num_filters):

        kernel = Kernels[i]

        fm = Convolution(Input, kernel)

        fm = ReLU(fm)

        output[:,:,i] = fm

    return output
    

I = cv.imread("./data/LR/Sat/image_t1_001.jpg")

padded = np.pad(
    I,
    ((1,1),(1,1),(0,0)),
    mode='constant'
)

num_filters = 10
kernel_size = 3
in_channels = 3

kernels = np.random.randn(
    num_filters,
    kernel_size,
    kernel_size,
    in_channels
)


ConvoLayer(padded,kernels)