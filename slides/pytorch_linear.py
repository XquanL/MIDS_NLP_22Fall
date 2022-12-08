"""Test binary classifier."""
# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from gen_simple import gen_simple


class BinaryLinear:
    """Binary linear NN classifier."""

    def __init__(self, X, Y, learning_rate=1e-2):
        """Initialize."""
        self.X = X  # leave x as numpy array
        self.Y = Y.reshape(
            -1, 1
        )  # -1 means allow numpy to figure out the rows, but we want 1 column
        self.Y_t = torch.FloatTensor(self.Y)  # turn labels into torch tensor
        self.n_input_dim = X.shape[1]  # 2 dimensions
        self.learning_rate = learning_rate

        # Build network
        self.weights = torch.randn(self.n_input_dim, 1, requires_grad=True)  # 2x1
        self.bias = torch.randn(1, requires_grad=True)  # a scalar
        # the function that we're trying to learn here is just a linear projection of those data
        # a linear classifier corresponds to a linear function
        # initialize these randomly to do gradient descent to identify what those weights and bias should be.

        # This just gives you is convenient shortcuts for these common pieces.
        # self.net = nn.Sequential(
        #     nn.Linear(self.n_input_dim, 1),
        #     nn.Sigmoid(),
        # )

        # binary cross entropy loss
        # self.loss_func = nn.BCELoss()

        ## this optimizer provides some utilities to do some of the things that we did here,
        # updating data based on the gradients, doing this backward, that sort of thing.
        # self.optimizer = torch.optim.Adam(
        #    self.net.parameters(),
        #    lr=self.learning_rate,
        # )

    def predict(self, X):
        """Predict."""
        # Function to generate predictions based on data
        X_t = torch.FloatTensor(X)
        return 1 / (1 + torch.exp(-(X_t @ self.weights) + self.bias))
        # sigmoid function is the activation function
        # what we're sticking into it is here it's x matrix multiplied by some weights.
        # This is a neural network. This is a simple as possible. Neural network has one layer of weight.
        # sigmoid is a good choice for binary classification. Right, because its outputs are between zero and one.

    def calculate_loss(self, y_hat):
        """Calculate loss."""
        return torch.sum(torch.abs(y_hat - self.Y_t))
        #  the sum of absolute differences between what we got and what we wanted to get.

    def update_network(self, y_hat):
        """Update weights."""
        loss = self.calculate_loss(y_hat)
        loss.backward()
        self.weights.data -= (
            self.weights.grad * self.learning_rate
        )  # we initialize the them randomly, and set requires_grad to true to tell pytorch to keep track of the gradients.
        self.bias.data -= self.bias.grad * self.learning_rate
        # the simple as possible gradient descent --- we take a step in the appropriate direction of this prespecified length(1e-2).

    @staticmethod
    def calculate_accuracy(y_hat_class, Y):
        """Calculate accuracy."""
        return np.sum(Y.reshape(-1, 1) == y_hat_class) / len(Y)

    def train(self, n_iters=1000):
        """Train network."""
        #  given the function that you have right now and your inputs predict what the output should be.
        # based on those outputs and their difference from the truth updates the function.
        for _ in range(n_iters):
            y_hat = self.predict(self.X)
            self.update_network(y_hat)

    def plot_testing_results(self, X_test, Y_test):
        """Plot testing results."""
        # Pass test data
        y_hat_test = self.predict(X_test)
        y_hat_test_class = np.where(y_hat_test < 0.5, 0, 1)
        print(
            "Test Accuracy {:.2f}%".format(
                self.calculate_accuracy(y_hat_test_class, Y_test) * 100
            )
        )

        # Plot the decision boundary
        # Determine grid range in x and y directions
        x_min, x_max = self.X[:, 0].min() - 0.1, self.X[:, 0].max() + 0.1
        y_min, y_max = self.X[:, 1].min() - 0.1, self.X[:, 1].max() + 0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(
            np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing)
        )

        # Concatenate data to match input
        data = np.hstack(
            (
                XX.ravel().reshape(-1, 1),
                YY.ravel().reshape(-1, 1),
            )
        )

        # Pass data to predict method
        db_prob = self.predict(data)

        clf = np.where(db_prob < 0.5, 0, 1)

        Z = clf.reshape(XX.shape)

        plt.figure(figsize=(12, 8))
        plt.contourf(
            XX,
            YY,
            Z,
            alpha=0.3,
        )
        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=Y_test,
        )
        plt.savefig("linear_regression.png")


def main():
    """Run experiment."""
    X, Y = gen_simple(500)

    # Split into test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
    )

    net = BinaryLinear(X_train, Y_train, learning_rate=1e-6)
    net.train()
    net.plot_testing_results(X_test, Y_test)


def train_test_split(X, Y, test_size=0.25):
    """Split into train and test sets."""
    N = X.shape[0]
    test_samples = np.random.choice(N, int(N * test_size), replace=False)
    train_samples = np.setdiff1d(np.arange(N), test_samples)
    X_test = X[test_samples]
    Y_test = Y[test_samples]
    X_train = X[train_samples]
    Y_train = Y[train_samples]
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    main()
