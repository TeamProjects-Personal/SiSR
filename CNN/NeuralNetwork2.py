import numpy as np

class NeuralNetwork:

    # -------------------------------------------------
    # INITIALIZE NETWORK
    # -------------------------------------------------
    def __init__(self, layer_sizes, learning_rate=0.1):

        self.learning_rate = learning_rate

        # Weight matrices
        weight_shapes = [
            (a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])
        ]

        
        self.weights = [
            np.random.randn(*s) * np.sqrt(1 / s[1])
            for s in weight_shapes
        ]

        # Bias vectors
        self.biases = [
            np.zeros((s, 1))
            for s in layer_sizes[1:]
        ]
        
        #self.load_model("Digits_Recongnization.npz");




    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):

        sig = self.activation(x)

        return sig * (1 - sig)


    def predict(self, a):

        for w, b in zip(self.weights, self.biases):

            z = w @ a + b

            a = self.activation(z)

        return a


    def forward(self, x):

        activations = [x]
        zs = []

        a = x

        for w, b in zip(self.weights, self.biases):

            z = w @ a + b

            zs.append(z)

            a = self.activation(z)

            activations.append(a)

        return activations, zs


    def cost(self, output, y):
        return np.mean((output - y) ** 2)

    # -------------------------------------------------
    # BACKPROPAGATION
    # -------------------------------------------------
    def backprop(self, x, y):

        # Forward pass
        activations, zs = self.forward(x)


        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]


        delta = (
            (activations[-1] - y)
            * self.activation_derivative(zs[-1])
        )

        # Gradients for last layer
        grad_w[-1] = delta @ activations[-2].T
        grad_b[-1] = delta


        for l in range(2, len(self.weights) + 1):

            z = zs[-l]

            sp = self.activation_derivative(z)

            delta = (self.weights[-l + 1].T @ delta) * sp

            grad_w[-l] = delta @ activations[-l - 1].T

            grad_b[-l] = delta


        for i in range(len(self.weights)):

            self.weights[i] -= self.learning_rate * grad_w[i]

            self.biases[i] -= self.learning_rate * grad_b[i]

    def train(self, X, Y, epochs):

        for epoch in range(epochs):

            for x, y in zip(X, Y):

                # Convert to column vectors
                x = np.array(x).reshape(-1, 1)
                y = np.array(y).reshape(-1, 1)

                self.backprop(x, y)

            # Print loss every 100 epochs
            if epoch % 100 == 0:

                total_loss = 0

                for x, y in zip(X, Y):

                    x = np.array(x).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)

                    pred = self.predict(x)

                    total_loss += self.cost(pred, y)

                print(f"Epoch {epoch} Loss = {total_loss:.4f}")
    
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

