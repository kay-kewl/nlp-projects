import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ElasticNetLogit(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with Elastic Net (L1 + L2) regularization.
    Implemented using Gradient Descent from scratch.
    """

    def __init__(
        self,
        beta=1.0,
        gamma=1.0,
        learning_rate=1e-3,
        tolerance=0.01,
        max_iter=1000,
        random_state=42,
    ):
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

    def _loss(self, X, y, w):
        epsilon = 1e-15
        sigma = self._sigmoid(X @ w)
        sigma = np.clip(sigma, epsilon, 1 - epsilon)
        data_loss = -np.sum(y * np.log(sigma) + (1 - y) * np.log(1 - sigma))
        reg_loss = self.gamma * np.linalg.norm(w, ord=1) + self.beta * np.sum(w**2)
        return data_loss + reg_loss

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack((ones, X))
        np.random.seed(self.random_state)
        self.w = np.random.uniform(-0.1, 0.1, X_aug.shape[1])

        for _ in range(self.max_iter):
            prev_w = self.w.copy()
            sigma = self._sigmoid(X_aug @ prev_w)
            grad = (
                X_aug.T @ (sigma - y)
                + self.gamma * np.sign(prev_w)
                + 2 * self.beta * prev_w
            )
            self.w = prev_w - self.learning_rate * grad

            self.loss_history.append(self._loss(X_aug, y, self.w))

            if np.linalg.norm(self.w - prev_w) < self.tolerance:
                break
        return self

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack((ones, X))
        return (X_aug @ self.w > 0).astype(int)
