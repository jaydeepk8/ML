import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_epoch=50, random_state=1):
        self.eta = eta
        self.n_epoch = n_epoch
        self.random_state = random_state

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return np.where(z >= 0.0, 1, 0)

    def predict(self, X):
        return self.activation(self.net_input(X))

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.0)
        self.errors_ = []

        for _ in range(self.n_epoch):
            n_errors = 0

            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                update = self.eta * error
                self.w_ += update * xi
                self.b_ += update
                n_errors += int(error != 0.0)

            self.errors_.append(n_errors)

        return self