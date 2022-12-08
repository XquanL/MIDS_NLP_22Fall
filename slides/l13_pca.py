"""Demonstrate PCA."""
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

MARKERS = ["o", "v", "+"]


class Dataset:
    """Dataset for classification."""

    def __init__(
        self, data: npt.NDArray[np.float_], labels: npt.NDArray[np.int_]
    ):
        """Initialize."""
        self.data = data
        self.labels = labels

    def plot(self):
        """Plot classification data."""
        for label in np.unique(self.labels):
            data_class = self.data[self.labels == label, :]
            x = data_class[:, 0]
            y = (
                data_class[:, 1]
                if data_class.shape[1] > 1
                else np.ones(data_class.shape)
            )
            plt.scatter(x, y)
        plt.axis("equal")

    def split(self, num_train: int):
        """Split into two datasets randomly."""
        N = self.data.shape[0]
        train_samples = np.random.choice(N, size=num_train, replace=False)
        test_samples = np.setdiff1d(np.arange(N), train_samples)
        return Dataset(
            self.data[train_samples, :], self.labels[train_samples]
        ), Dataset(self.data[test_samples, :], self.labels[test_samples])

    def percent_correct(self):
        """Compute percent correct."""
        return np.sum(self.data.ravel() == self.labels) / self.labels.size


class KNN:
    """K-nearest neighbors classifier."""

    def __init__(self, k: int):
        """Initialize."""
        self.k = k

    def train(self, dataset: Dataset):
        """Train."""
        self.X = dataset.data
        self.Y = dataset.labels
        return self

    def compute_distances(self, X):
        """Compute distances between test and train points."""
        return (
            np.add.reduce(
                [
                    (X[:, [idx]] - self.X[:, [idx]].T) ** 2
                    for idx in range(X.shape[1])
                ]
            )
            ** 0.5
        )

    def nearby(self, X):
        """Compute ordered list of nearby labels."""
        distances = self.compute_distances(X)
        idx = np.argsort(distances, axis=1)
        distances = np.sort(distances, axis=1)
        return self.Y[idx], distances

    def apply(self, dataset: Dataset):
        """Find the most-common label in the top k."""
        modes = []
        labels, _ = self.nearby(dataset.data)
        for ys in labels:
            p = 1 / self.k
            modes.append(round(np.sum(ys[: self.k] * p)))
        return Dataset(np.array([modes]).T, dataset.labels)


class FLD:
    """Fisher's linear discriminant."""

    def __init__(self):
        """Initialize."""
        self.w = None

    def train(self, dataset: Dataset):
        """Train."""
        data_0 = dataset.data[dataset.labels == 0, :]
        mu_0 = np.mean(data_0, axis=0, keepdims=True).T
        Sigma_0 = np.cov(data_0.T)
        data_1 = dataset.data[dataset.labels == 1, :]
        mu_1 = np.mean(data_1, axis=0, keepdims=True).T
        Sigma_1 = np.cov(data_1.T)
        self.w = np.linalg.solve(Sigma_1 + Sigma_0, mu_1 - mu_0)
        return self

    def apply(self, dataset: Dataset):
        """Apply."""
        if self.w is None:
            raise ValueError("FLD must be trained first.")
        return Dataset(dataset.data @ self.w, dataset.labels)


def best_correct(dataset: Dataset):
    """Compute the best possible percent correct."""
    max_correct = 0.0
    for threshold in dataset.data.ravel():
        labels_below = dataset.labels[dataset.data.ravel() < threshold]
        labels_above = dataset.labels[dataset.data.ravel() >= threshold]
        correct = (
            sum(labels_below == 0) + sum(labels_above == 1)
        ) / dataset.data.size
        if correct > max_correct:
            max_correct = correct
    return max_correct


def random_rotation(ndims: int) -> npt.NDArray[np.float_]:
    """Generate random rotation matrix."""
    X = np.random.standard_normal(size=(ndims, ndims))
    Q, _ = np.linalg.qr(X)
    return Q


def mean_vec(idx: int, ndims: int) -> npt.NDArray[np.float_]:
    """Generate mean vector."""
    mean = np.zeros((ndims,))
    mean[0] = idx
    return mean


def cov_mat(ndims: int, R: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Generate covariance matrix."""
    noise_var = 0.1
    onehot = np.zeros((ndims,)) + noise_var
    onehot[0] = 1.0
    return np.diag(onehot) @ R


def generate_onehot(D: int, indices: npt.ArrayLike):
    """Generate one-hot vectors from indices."""
    indices = np.array(indices)
    data = np.zeros((indices.size, D))
    for row_idx, idx in enumerate(indices):
        data[row_idx][idx] = 1.0
    return data


def generate_data(
    num_dims: int,
    num_classes: int,
    num_samples: int,
    R: npt.NDArray[np.float_],
):
    """Generate data."""
    scale = 10
    data = np.empty((0, num_dims))
    labels = np.empty((0,), dtype=np.int_)
    for idx_class in range(num_classes):
        data_class = np.round(
            (
                np.random.multivariate_normal(
                    mean_vec(idx_class, num_dims),
                    0.1 * np.eye(num_dims),
                    size=(num_samples,),
                )
                @ R
            )
            * scale
        )
        data = np.vstack((data, data_class))
        labels_class = np.ones((num_samples,), dtype=np.int_) * idx_class
        labels = np.concatenate((labels, labels_class))
    # unique_rows, unique_inverse = np.unique(data, axis=0, return_inverse=True)
    # data = generate_onehot(unique_rows.shape[0], unique_inverse)
    # print(unique_rows, unique_inverse, data)
    return Dataset(data, labels)


def plot_correct_vs_dims():
    """Plot correctness as a function of the number of predictors."""
    num_classes = 2
    num_samples = 20
    num_dims = np.arange(2, 20)
    replicates = 10
    correctness = []
    for D in num_dims:
        correctness_D = []
        for _ in range(replicates):
            R = random_rotation(D)
            dataset = generate_data(D, num_classes, num_samples + 100, R)
            dataset_train, dataset_test = dataset.split(num_samples)

            knn = KNN(5).train(dataset_train)
            guess_dataset_test = knn.apply(dataset_test)
            correctness_D.append(guess_dataset_test.percent_correct())

            # fld = FLD().train(dataset_train)
            # proj_dataset_test = fld.apply(dataset_test)
            # correctness_D.append(best_correct(proj_dataset_test))

        correctness.append(np.mean(correctness_D))
    plt.plot(num_dims, correctness)
    plt.xlabel("number of dimensions")
    plt.ylabel("percent correct")


def main():
    """Demonstrate PCA."""
    num_classes = 2
    num_dims = 2
    num_samples = 10
    R = random_rotation(num_dims)

    # example data (2D slice)
    dataset_train = generate_data(num_dims, num_classes, num_samples, R)
    dataset_train.plot()
    plt.show()

    fld = FLD().train(dataset_train)

    # example projected data
    proj_dataset_train = fld.apply(dataset_train)
    proj_dataset_train.plot()
    plt.show()

    # run experiment
    plot_correct_vs_dims()
    plt.show()


if __name__ == "__main__":
    main()
