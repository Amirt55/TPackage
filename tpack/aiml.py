import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import resample


def generate_data(n=50, noise=5.0, plot=False):
  np.random.seed(42)
  X = np.linspace(-10, 10, n)
  noise = np.random.randn(n) * noise
  y_function = input("Enter the function in terms of X (e.g., 'X**2 + 3*X + 2'): ")
  y = eval(y_function)
  y += noise
  if plot:
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.title("Generated Data (Univariate)")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.show()
  return X, y


def polynomial_features(X, degree):
  X_poly = np.c_[np.ones(len(X))]
  for i in range(1, degree + 1):
    X_poly = np.c_[X_poly, X**i]
  return X_poly


def polynomial_regression(X, y, degree):
  X_poly = polynomial_features(X, degree)
  # Closed-form solution: w = (X'^T * X')^-1 * X'^T * y
  w = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
  return w


def linear_regression_closed_form(X, y):
  # Adding bias term (x_0 = 1) to input vector X
  X_b = np.c_[np.ones((len(X), 1)), X]  # X_b is now the full input vector with bias term
  # Closed-form solution: w = (X^T * X)^-1 * X^T * y
  w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
  return w


def gradient_descent(X, y, w, alpha, num_iters, h_w, cost_function):
  m = len(X)
  cost_history = []
  w_history = [w.copy()]
  for i in range(num_iters):
    # updates
    gradient_w0 = np.sum(h_w(X, w) - y) / m
    gradient_w1 = np.sum((h_w(X, w) - y) * X) / m
    w[0] -= alpha * gradient_w0
    w[1] -= alpha * gradient_w1
    cost_history.append(cost_function(X, y, w))
    w_history.append(w.copy())  # Store a copy of w, not the reference
  return w, cost_history, w_history


class Perceptron:
  def __init__(self, learning_rate=0.01, n_epochs=1000):
      self.learning_rate = learning_rate
      self.n_epochs = n_epochs
      self.weights = None
      self.bias = None
      self.errors_ = []  # storing the number of misclassifications in each epoch

  def fit(self, X, y):
    """
    Train the Perceptron model on the provided data.

    Parameters:
    X : array-like, shape = [n_samples, n_features]
        Training vectors.
    y : array-like, shape = [n_samples]
        Target values. Must be +1 or -1.
    """
    n_samples, n_features = X.shape
    # starting weights and bias equal zeros
    self.weights = np.zeros(n_features)
    self.bias = 0.0
    for epoch in range(self.n_epochs):
      errors = 0
      for idx in range(n_samples):
        linear_output = np.dot(X[idx], self.weights) + self.bias  # w^T x + b
        y_pred = self._unit_step(linear_output)
        if y[idx] != y_pred: # misclassfied
          update = self.learning_rate * y[idx]
          self.weights += update * X[idx]
          self.bias += update
          errors += 1
      self.errors_.append(errors)
      # if no errors, convergence achieved
      if errors == 0:
        print(f"Converged after {epoch+1} epochs")
        break

  def predict(self, X):
    """
    Predict class labels for samples in X.

    Parameters:
    X : array-like, shape = [n_samples, n_features]

    Returns:
    array, shape = [n_samples]
        Predicted class labels.
    """
    linear_output = np.dot(X, self.weights) + self.bias
    return self._unit_step(linear_output)

  def _unit_step(self, x):
    return np.where(x >= 0, 1, -1)

  
class KNNClassifier:
  def __init__(self, k=3, distance_func=None):
    self.k = k
    if distance_func is None:
      self.distance_func = self._euclidean_distance
    else:
      self.distance_func = distance_func

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def _euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

  def _manhattan_distance(self, x1, x2):
    return np.sum(np.abs(x1 - x2))

  def _minkowski_distance(self, x1, x2, p=3):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p)

  def predict(self, X):
    predictions = []
    for index, x in enumerate(X):
      distances = [self.distance_func(x, x_train) for x_train in self.X_train]
      # indices of the k nearest neighbors
      k_indices = np.argsort(distances)[:self.k]
      # labels of the k nearest neighbors
      k_neighbor_labels = self.y_train[k_indices]
      # majority vote
      counts = np.bincount(k_neighbor_labels.astype(int))
      predicted_label = np.argmax(counts)
      predictions.append(predicted_label)
    return np.array(predictions)


class DecisionStump:
  """
  A decision stump classifier for multi-class classification problems (depth = 1).
  """

  def __init__(self):
    self.feature = None
    self.threshold = None
    self.value_left = None
    self.value_right = None

  def fit(self, X, y):
    """
    Fits a decision stump to the dataset (X, y).
    """
    best_gain = -1
    for feature_index in range(X.shape[1]):
      thresholds = np.unique(X[:, feature_index])
      for threshold in thresholds:
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        left_y, right_y = y[left_mask], y[right_mask]
        if len(left_y) and len(right_y):
          left_weight = len(left_y) / len(y)
          right_weight = 1 - left_weight
          gain = self._entropy(y) - (left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y))
          if gain > best_gain:
            best_gain = gain
            self.feature = feature_index
            self.threshold = threshold
            self.value_left = np.bincount(left_y).argmax()
            self.value_right = np.bincount(right_y).argmax()

  def predict(self, X):
    """
    Predicts class labels for samples in X.
    """
    return np.where(X[:, self.feature] <= self.threshold, self.value_left, self.value_right)

  def _entropy(self, y):
    """
    Computes entropy for a set of labels.
    """
    proportions = np.bincount(y) / len(y)
    return -np.sum([p * np.log2(p) for p in proportions if p > 0])


class RandomForest:
  """
  A random forest classifier for multi-class classification problems (using decision stumps with depth 1).
  """
  def __init__(self, n_trees=7):
    self.n_trees = n_trees
    self.trees = []

  def fit(self, X, y):
    """
    Fits a random forest to the dataset (X, y).
    """
    self.trees = []
    for _ in range(self.n_trees):
      stump = DecisionStump()
      X_sample, y_sample = self._bootstrap_samples(X, y)
      stump.fit(X_sample, y_sample)
      self.trees.append(stump)

  def predict(self, X):
    """
    Predicts class labels for samples in X.
    """
    stump_predictions = np.array([stump.predict(X) for stump in self.trees])
    return self._majority_vote(stump_predictions)

  def _bootstrap_samples(self, X, y):
    """
    Applies bootstrap resampling to the dataset.
    """
    return resample(X, y, n_samples=len(X), replace=True)

  def _majority_vote(self, predictions):
    """
    Returns the majority vote of the predictions.
    """
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)


def Visualize(X, centroids, labels):
  plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
  plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
  plt.show()


def kmeans(X, k, iterations=100):
  centroids = X[np.random.choice(X.shape[0], k, replace=False)]
  for _ in range(iterations):
    labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)
    Visualize(X, centroids, labels)
    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
    if np.all(centroids == new_centroids): break
    centroids = new_centroids
  return centroids, labels


def pca(X, num_components):
  X_meaned = X - np.mean(X, axis=0)
  covariance_matrix = np.cov(X_meaned, rowvar=False)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
  sorted_index = np.argsort(eigenvalues)[::-1]
  sorted_eigenvectors = eigenvectors[:, sorted_index]
  eigenvector_subset = sorted_eigenvectors[:, :num_components]
  X_reduced = np.dot(X_meaned, eigenvector_subset)
  return X_reduced, eigenvector_subset
