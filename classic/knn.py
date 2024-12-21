import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


class KNN:
    def __init__(self, k=3, metric='euclidean', weighted=True):
        """
        Initialize the KNN model.

        Parameters:
        - k: int, number of neighbors to consider.
        - metric: str, distance metric ('euclidean' or 'manhattan').
        - weighted: bool, whether to use weighted voting.
        """
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fit the training data.

        Parameters:
        - X: array-like, training data points.
        - y: array-like, labels for training data.
        """
        self.X_train = self.scaler.fit_transform(X)  # Scale training data
        self.y_train = np.array(y)

    def _compute_distance(self, point1, point2):
        """
        Compute distance between two points based on the chosen metric.
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError(
                "Unsupported metric. Use 'euclidean' or 'manhattan'.")

    def _get_neighbors(self, test_point):
        """
        Find the k-nearest neighbors for a given test point.
        """
        distances = [self._compute_distance(
            test_point, train_point) for train_point in self.X_train]
        neighbors_indices = np.argsort(distances)[:self.k]
        return neighbors_indices, distances

    def predict(self, X_test):
        """
        Predict the labels for the test data.

        Parameters:
        - X_test: array-like, test data points.

        Returns:
        - predictions: list of predicted labels for X_test.
        """
        X_test_scaled = self.scaler.transform(X_test)  # Scale test data
        predictions = []

        for test_point in X_test_scaled:
            neighbors_indices, distances = self._get_neighbors(test_point)
            neighbor_labels = self.y_train[neighbors_indices]

            if self.weighted:
                # Weighted voting: inversely proportional to distance
                # Avoid division by zero
                weights = 1 / (np.array(distances)[neighbors_indices] + 1e-5)
                weighted_votes = {}
                for label, weight in zip(neighbor_labels, weights):
                    weighted_votes[label] = weighted_votes.get(
                        label, 0) + weight
                prediction = max(weighted_votes, key=weighted_votes.get)
            else:
                # Majority voting
                prediction = Counter(neighbor_labels).most_common(1)[0][0]

            predictions.append(prediction)
        return predictions


# Example usage
if __name__ == "__main__":
    # Example dataset
    train_data = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7]])
    train_labels = np.array(['A', 'A', 'A', 'B', 'B'])
    test_data = np.array([[2, 2], [6, 6]])

    # Initialize and train the model
    knn = KNN(k=3, metric='euclidean', weighted=True)
    knn.fit(train_data, train_labels)

    # Predict
    predictions = knn.predict(test_data)
    print("Predictions:", predictions)
