import numpy as np
from collections import Counter


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def knn_predict(train_data, train_labels, test_data, k):
    """
    Perform k-Nearest Neighbors prediction.

    Parameters:
    - train_data: list or array-like, training data points.
    - train_labels: list or array-like, labels for training data.
    - test_data: list or array-like, test data points to classify.
    - k: int, number of neighbors to consider.

    Returns:
    - predictions: list of predicted labels for test_data.
    """
    predictions = []

    for test_point in test_data:
        # Compute distances from the test point to all training points
        distances = [euclidean_distance(test_point, train_point)
                     for train_point in train_data]

        # Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:k]

        # Get the labels of the k nearest neighbors
        k_neighbors_labels = [train_labels[i] for i in k_indices]

        # Determine the majority label (classification)
        most_common = Counter(k_neighbors_labels).most_common(1)
        predictions.append(most_common[0][0])

    return predictions


# Example usage
if __name__ == "__main__":
    print("start")
    # Example dataset
    train_data = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 7]]
    train_labels = ['A', 'A', 'A', 'B', 'B']
    test_data = [[2, 2], [6, 6]]

    # Set k
    k = 3

    # Predict
    predictions = knn_predict(train_data, train_labels, test_data, k)
    print("Predictions:", predictions)
