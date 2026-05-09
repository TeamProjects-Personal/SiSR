import numpy as np
import SRCNN 

class ConvolutionNeuralNetwork:

    def he_init(self,kh, kw, cin, cout):
        scale = np.sqrt(2.0 / cin)
        return np.random.randn(kh, kw, cin, cout) * scale

    def __init__(self, num_filter, learning_rate=0.1):

        self.learning_rate = learning_rate

        # Bias vectors
        self.biases = np.zeros(num_filter)

        
        num_filters = 10
        kernel_size = 3
        in_channels = 3

        self.kernels = np.random.randn(
            num_filters,
            kernel_size,
            kernel_size,
            in_channels
        )

        

    def activation(self, x):
        return np.maximum(0,x)

    def activation_derivative(self, x):

        sig = self.activation(x)

        return sig * (1 - sig)


    def predict(self, X,K):

        for w, b in zip(self.weights, self.biases):

            z = SRCNN.Convolution(X,K) + b

            a = self.activation(z)

        return a

    def compute_grad_kernel(x, delta, kH, kW, C_in, C_out):

        H, W, _ = x.shape
        H_out, W_out, _ = delta.shape

        grad_k = np.zeros((C_out, kH, kW, C_in))

        for f in range(C_out):
            for i in range(H_out):
                for j in range(W_out):

                    for cin in range(C_in):
                        grad_k[f, :, :, cin] += (
                            x[i:i+kH, j:j+kW, cin] * delta[i, j, f]
                        )

        return grad_k

    def cost(self, output, y):
        return np.mean((output - y) ** 2)

    def backprop(self, x,k, y):

        # Forward pass
        activations,zs = SRCNN.ConvoLayer(x,k,self.biases)

        A_L = activations
        Z_L = zs

        H,W,n_filter = activations.shape
        kn,kh,kw,kc = k.shape
        delta = (A_L - y)*self.activation_derivative(Z_L)

        grad_b = np.zeros(n_filter)
        grad_kernel = self.compute_grad_kernel(x,delta,kh,kw,n_filter,kc)
        for l in reversed(range(n_filter)):

            
            grad_b[l] = np.sum(delta[:,:,l])

            if(l>0):
                delta =  SRCNN.conv_transpose(delta,k)
                delta *= self.activation_derivative(zs[:,:,l-1])

                



    def train(self, X_train, Y_train, epochs):

        for epoch in range(epochs):

            total_loss = 0

            for x, y in zip(X_train, Y_train):


                activations, zs = SRCNN.ConvoLayer(x, self.kernels, self.biases)

                loss = np.mean((activations - y) ** 2)
                total_loss += loss


                grad_kernel, grad_b = self.backprop(x, self.kernels, y)


                self.kernels -= self.learning_rate * grad_kernel
                self.biases  -= self.learning_rate * grad_b

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(X_train)}")
    
    def save_model(self, filename):

        np.savez(
            filename,
            weights=np.array(self.weights, dtype=object),
            biases=np.array(self.biases, dtype=object)
        )
    def load_model(self, filename):

        data = np.load(filename, allow_pickle=True)

        self.weights = list(data["weights"])

        self.biases = list(data["biases"])

def pad_image(img, pad_h, pad_w):
    return np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant',
        constant_values=0
    )