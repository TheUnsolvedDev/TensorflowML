# Logistic Regression

Logistic regression is a statistical method used to model the probability of a binary outcome based on one or more independent variables. It is widely used for classification problems. Here is an overview of the mathematics behind logistic regression:

## 1. Model Representation

For a binary logistic regression with one independent variable, the model is represented as:
$$ \text{logit}(P(Y=1)) = \log\left(\frac{P(Y=1)}{1 - P(Y=1)}\right) = \beta_0 + \beta_1 x $$
where:
- $ P(Y=1) $ is the probability that the dependent variable $ Y $ equals 1.
- $ x $ is the independent variable.
- $ \beta_0 $ is the intercept.
- $ \beta_1 $ is the coefficient for the independent variable.

The logistic function (sigmoid function) is used to model the probability:
$$ P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}} $$

For multiple logistic regression with $ p $ independent variables, the model extends to:
$$ P(Y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p)}} $$

## 2. Objective Function

The goal is to find the values of $ \beta_0, \beta_1, \ldots, \beta_p $ that maximize the likelihood function. The likelihood function for $ n $ observations is:
$$ L(\beta) = \prod_{i=1}^n P(y_i)^{y_i} (1 - P(y_i))^{1 - y_i} $$
where $ y_i $ is the observed outcome (0 or 1).

The log-likelihood function, which is easier to maximize, is:
$$ \ell(\beta) = \sum_{i=1}^n \left[ y_i \log(P(y_i)) + (1 - y_i) \log(1 - P(y_i)) \right] $$

## 3. Solution

The parameters $ \beta_0, \beta_1, \ldots, \beta_p $ are estimated using maximum likelihood estimation (MLE). This involves finding the values that maximize the log-likelihood function. Numerical optimization algorithms, such as gradient descent or iteratively reweighted least squares (IRLS), are commonly used.

## 4. Assumptions

Logistic regression relies on several key assumptions:
- **Linearity of the logit**: The logit (log-odds) of the outcome is a linear combination of the independent variables.
- **Independence**: Observations are independent of each other.
- **No multicollinearity**: Independent variables are not highly correlated.

## 5. Evaluation Metrics

### Binary Accuracy

Binary accuracy measures the proportion of correct predictions:
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
where:
- $ TP $ = True Positives
- $ TN $ = True Negatives
- $ FP $ = False Positives
- $ FN $ = False Negatives

### Precision

Precision measures the proportion of positive predictions that are actually correct:
$$ \text{Precision} = \frac{TP}{TP + FP} $$

### Recall (Sensitivity)

Recall measures the proportion of actual positives that are correctly identified:
$$ \text{Recall} = \frac{TP}{TP + FN} $$

### F1 Score

The F1 score is the harmonic mean of precision and recall:
$$ \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

### ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve plots the true positive rate (recall) against the false positive rate (1 - specificity). The AUC (Area Under the Curve) measures the overall performance of the model:
- **AUC = 1**: Perfect model
- **AUC = 0.5**: Model performs no better than random chance

### Log Loss

Log loss (logarithmic loss) measures the accuracy of the classifier's probability estimates:
$$ \text{Log Loss} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(P(y_i)) + (1 - y_i) \log(1 - P(y_i)) \right] $$

These metrics provide a comprehensive view of the model's performance in various aspects, allowing for a thorough evaluation of logistic regression models.