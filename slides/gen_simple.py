"""Generate simple data."""
# pylint: disable=invalid-name
import matplotlib.pyplot as plt
import numpy as np


def gen_simple(nobs_per=50):
    """Generate simple data."""
    X = np.vstack((
        np.random.randn(nobs_per, 2) * 0.25 + [[0, 0]],
        np.random.randn(nobs_per, 2) * 0.25 + [[1, 0]],
    ))
    Y = np.vstack((
        np.zeros((nobs_per, 1)),
        np.ones((nobs_per, 1)),
    ))
    return X, Y


def main():
    """Plot simple data."""
    X, Y = gen_simple()
    for y in np.unique(Y):
        plt.plot(X[Y[:, 0] == y, 0], X[Y[:, 0] == y, 1], 'o')
    plt.savefig("gen-simple-data.png")


if __name__ == "__main__":
    main()
