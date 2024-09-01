# K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for classification and regression. It is widely used due to its simplicity and effectiveness in various applications. Here is an overview of the mathematics behind KNN:

## 1. Model Representation

KNN does not assume any underlying probability distribution of the data. Instead, it relies on the distances between data points to make predictions.

### Classification

For a given test data point \( x \), KNN classifies it by looking at the \( k \) nearest neighbors (points) in the training data and assigning the most common class among them. 

### Regression

For a given test data point \( x \), KNN predicts the value by averaging the values of the \( k \) nearest neighbors in the training data.

## 2. Distance Metrics

The distance between two data points is a crucial part of KNN. Common distance metrics include:

### Euclidean Distance

For two points \( x = (x_1, x_2, \ldots, x_n) \) and \( y = (y_1, y_2, \ldots, y_n) \), the Euclidean distance is:
$$ d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2} $$

### Manhattan Distance

The Manhattan distance (also known as L1 norm) is:
$$ d(x, y) = \sum_{i=1}^n |x_i - y_i| $$

### Minkowski Distance

The Minkowski distance generalizes both Euclidean and Manhattan distances:
$$ d(x, y) = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{1/p} $$
where \( p \) is a parameter.

## 3. Algorithm

### Classification

1. **Choose \( k \)**: Decide the number of neighbors \( k \).
2. **Compute Distances**: For a test point \( x \), compute the distance from \( x \) to all points in the training set.
3. **Find Nearest Neighbors**: Identify the \( k \) points in the training set that are closest to \( x \).
4. **Vote**: Assign \( x \) the class that is most frequent among the \( k \) nearest neighbors.

### Regression

1. **Choose \( k \)**: Decide the number of neighbors \( k \).
2. **Compute Distances**: For a test point \( x \), compute the distance from \( x \) to all points in the training set.
3. **Find Nearest Neighbors**: Identify the \( k \) points in the training set that are closest to \( x \).
4. **Average**: Predict the value of \( x \) as the average of the values of the \( k \) nearest neighbors.

## 4. Assumptions

KNN makes several implicit assumptions:
- **Distance Metric**: The choice of distance metric is appropriate for the data.
- **Locality**: Points that are close to each other are more likely to have similar values.
- **Relevance of Neighbors**: The nearest neighbors are the most relevant for predicting the outcome.

## 5. Evaluation Metrics

### Classification

#### Accuracy

Accuracy measures the proportion of correct predictions:
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

#### Precision

Precision measures the proportion of positive predictions that are actually correct:
$$ \text{Precision} = \frac{TP}{TP + FP} $$

#### Recall (Sensitivity)

Recall measures the proportion of actual positives that are correctly identified:
$$ \text{Recall} = \frac{TP}{TP + FN} $$

#### F1 Score

The F1 score is the harmonic mean of precision and recall:
$$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

#### ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC (Area Under the Curve) measures the overall performance of the model:
- **AUC = 1**: Perfect model
- **AUC = 0.5**: Model performs no better than random chance
