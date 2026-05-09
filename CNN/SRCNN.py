import numpy as np
import cv2 as cv

def ExtractRegion(Image,r,c,kh,kw):
    return Image[r:r+kh, c:c+kw,:]

def ReLU(Image):
    return np.maximum(0,Image)

def conv_transpose(X, dZ, K):

    H, W, C_in = X.shape
    n_filters, kH, kW, _ = K.shape

    H_out = H - kH + 1
    W_out = W - kW + 1

    dX = np.zeros_like(X)

    for f in range(n_filters):
        for i in range(H_out):
            for j in range(W_out):

                grad = dZ[i, j, f]

                for cin in range(C_in):
                    for ki in range(kH):
                        for kj in range(kW):

                            dX[i + ki, j + kj, cin] += (
                                grad * K[f, ki, kj, cin]
                            )

    return dX

def Convolution(Image,kernel):
    h,w,channels = Image.shape
    kh,kw,kc = kernel.shape

    if(kc != channels):
       raise ValueError("Kernel channels must match image channels")

    out_h = h - kh + 1
    out_w = w - kw + 1

    inter_feature_map = np.zeros((out_h, out_w))
 
    for i in range(out_h):
       for j in range(out_w):

            r = i
            c = j
            region =np.array(ExtractRegion(Image,r,c,kh,kw))
            inter_feature_map[i,j] = np.sum(region * kernel)
    
    return np.array(inter_feature_map)

def ConvoLayer(Input,Kernels,Biases):
    H, W, _ = Input.shape
    num_filters,kH, kW, c_out = Kernels.shape

    out_H = H - kH + 1
    out_W = W - kW + 1
    
    Z_cache  = np.zeros((out_H,out_W,num_filters))
    A_cache  = np.zeros((out_H,out_W,num_filters))
    
    for i in range(num_filters):

        kernel = Kernels[i]
        bias = Biases[i]

        Z = Convolution(Input, kernel)
        Z = Z + bias  # scalar per filter

        A = ReLU(Z) 

        Z_cache[:,:,i] = Z
        A_cache[:,:,i] = A


    return Z_cache,A_cache
    
def pad_image(img, pad_h, pad_w):
    return np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )

