# Mathematics Behind Linear Regression

Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. The basic idea is to find the best-fitting line (or hyperplane in higher dimensions) that describes this relationship. Here is an overview of the mathematics behind linear regression:

## 1. Model Representation

For a simple linear regression with one independent variable, the model is represented as:
$$ y = \beta_0 + \beta_1 x + \epsilon $$
where:
- $ y $ is the dependent variable.
- $ x $ is the independent variable.
- $ \beta_0 $ is the intercept of the regression line.
- $ \beta_1 $ is the slope of the regression line.
- $ \epsilon $ is the error term (the difference between the observed and predicted values).

For multiple linear regression with $ p $ independent variables, the model extends to:
$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \epsilon $$

## 2. Objective Function

The goal is to find the values of $ \beta_0, \beta_1, \ldots, \beta_p $ that minimize the sum of squared errors (SSE):
$$ SSE = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}))^2 $$
where $ n $ is the number of observations, $ y_i $ is the observed value, and $ \hat{y}_i $ is the predicted value.

## 3. Normal Equations

To minimize the MSE, we take the partial derivatives of SSE with respect to each $ \beta $ and set them to zero. This results in a system of linear equations known as the normal equations. For multiple linear regression, the normal equations can be written in matrix form as:
$$ \mathbf{X}^\top \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^\top \mathbf{y} $$
where:
- $ \mathbf{X} $ is the design matrix, including a column of ones for the intercept.
- $ \boldsymbol{\beta} $ is the vector of coefficients $[ \beta_0, \beta_1, \ldots, \beta_p ]^\top $.
- $ \mathbf{y} $ is the vector of observed values.

## 4. Solution

The solution to the normal equations is:
$$ \boldsymbol{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} $$
This gives the values of the coefficients that minimize the SSE.

## 5. Assumptions

Linear regression relies on several key assumptions:
- **Linearity**: The relationship between the dependent and independent variables is linear.
- **Independence**: Observations are independent of each other.
- **Homoscedasticity**: The variance of the error terms is constant across all levels of the independent variables.
- **Normality**: The error terms are normally distributed.

## 6. Goodness of Fit

The goodness of fit of the model can be assessed using the coefficient of determination, $ R^2 $:
$$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $$
where $ SS_{res} $ is the residual sum of squares, and $ SS_{tot} $ is the total sum of squares. $ R^2 $ indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## 7. Inference

After estimating the coefficients, we can make inferences about the population parameters:
- **Confidence Intervals**: Calculate the range within which the true coefficient values are likely to fall.
- **Hypothesis Testing**: Test if the coefficients are significantly different from zero using t-tests.

These steps provide the foundation for understanding and applying linear regression in various statistical and machine learning contexts.