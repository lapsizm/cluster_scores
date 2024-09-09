## Overview

This project provides two functions for evaluating clustering results: `silhouette_score` and `bcubed_score`. These functions are designed to help assess the quality of clustering algorithms in machine learning.

### Functions

#### `silhouette_score(x, labels)`

Calculates the silhouette score for a given clustering.

- **Parameters:**
  - `x` (np.ndarray): A 2D array where each row represents an individual data point.
  - `labels` (np.ndarray): A 1D array of cluster labels corresponding to each data point in `x`.

- **Returns:**
  - `float`: The silhouette score, which measures how similar a data point is to its own cluster compared to other clusters. The score ranges from -1 to 1, where a higher value indicates a better clustering result.

- **Description:**
  The silhouette score is computed based on the average distance between each point and all other points in the same cluster (cohesion) and the average distance between each point and the nearest cluster (separation). It provides a measure of how well each point is clustered.

#### `bcubed_score(true_labels, predicted_labels)`

Calculates the B-Cubed score for evaluating clustering or classification results.

- **Parameters:**
  - `true_labels` (np.ndarray): A 1D array of true labels for the data points.
  - `predicted_labels` (np.ndarray): A 1D array of predicted labels for the data points.

- **Returns:**
  - `float`: The B-Cubed score, which is the harmonic mean of precision and recall. This score evaluates how well the predicted labels match the true labels by considering individual data points.

- **Description:**
  The B-Cubed score is calculated by comparing the predicted labels with the true labels. It measures the precision and recall of each label and then combines these measures to provide a single score. This metric is useful for evaluating clustering results, especially in tasks where the number of clusters and their composition are important.

## Dependencies

- `numpy`: For numerical operations and handling arrays.
- `sklearn`: For distance calculations using the Euclidean metric.

## Installation

You can install the necessary dependencies using pip:

```bash
pip install numpy scikit-learn
```

## Usage

Here is a basic example of how to use the provided functions:

```python
import numpy as np
from your_module import silhouette_score, bcubed_score  # Replace 'your_module' with the actual module name

# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
labels = np.array([0, 0, 1, 1])

# Calculate silhouette score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score}")

# Example true and predicted labels
true_labels = np.array([0, 0, 1, 1])
predicted_labels = np.array([0, 1, 1, 1])

# Calculate B-Cubed score
bcubed_sc = bcubed_score(true_labels, predicted_labels)
print(f"B-Cubed Score: {bcubed_sc}")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.

## Contact

For questions or issues, please contact [murygin_280702@mail.ru](mailto:murygin_280702@mail.ru).

---

Feel free to modify and expand this README to better fit your project's needs!
