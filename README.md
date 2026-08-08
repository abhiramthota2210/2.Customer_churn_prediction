````markdown
# Telco Customer Churn Prediction — Logistic Regression From Scratch

A machine learning project that predicts whether a telecom customer is likely to churn using **Logistic Regression implemented completely from scratch with NumPy**.

The prediction pipeline does **not use Scikit-learn's LogisticRegression** or any other pre-built prediction algorithm.

## Project Overview

Customer churn is a major problem for subscription-based businesses. Being able to identify customers who are likely to leave allows a company to take preventive actions.

This project uses the **Telco Customer Churn** dataset to:

- Clean and preprocess customer data
- Convert categorical features into numerical features
- Scale numerical features
- Implement Logistic Regression from scratch
- Train the model using Gradient Descent
- Evaluate predictions using classification metrics
- Analyze model performance

## Dataset

The project uses the **Telco Customer Churn** dataset.

Each row represents a customer and contains information such as:

- Gender
- Senior citizen status
- Partner
- Dependents
- Tenure
- Phone service
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn

### Target Variable

`Churn`

| Value | Meaning |
|---|---|
| 1 | Customer churned |
| 0 | Customer did not churn |

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn

Scikit-learn was used only for preprocessing/evaluation utilities where applicable, **not for the prediction model itself**.

## Project Pipeline

```text
Raw Dataset
     |
     v
Data Cleaning
     |
     v
Remove Customer ID
     |
     v
Categorical Encoding
     |
     v
One-Hot Encoding
     |
     v
Train / Test Split
     |
     v
Feature Scaling
     |
     v
Initialize Parameters
     |
     v
Linear Model
     |
     v
Sigmoid Function
     |
     v
Binary Cross-Entropy Cost
     |
     v
Gradient Descent
     |
     v
Trained Logistic Regression Model
     |
     v
Predictions
     |
     v
Model Evaluation
````

## Logistic Regression

The model first calculates a linear combination of the input features:

$$
z = X\theta + b
$$

where:

* `X` = input features
* `θ` = model weights
* `b` = bias
* `z` = linear model output

The output is then passed through the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This converts the output into a probability between 0 and 1.

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

The prediction is then obtained using a threshold:

```python
y_pred = (probability >= 0.5).astype(int)
```

## Cost Function

Binary Cross-Entropy is used as the loss function:

$$
J(\theta,b)
===========

-\frac{1}{m}
\sum_{i=1}^{m}
[
y_i\log(\hat{y_i})
+
(1-y_i)\log(1-\hat{y_i})
]
$$

Implementation:

```python
def compute_cost(y, y_pred):
    m = len(y)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    loss = -(y * np.log(y_pred)
             + (1 - y) * np.log(1 - y_pred))

    return (1 / m) * np.sum(loss)
```

## Gradient Descent

The gradients used to update the model parameters are:

$$
\frac{\partial J}{\partial \theta}
==================================

\frac{1}{m}X^T(\hat{y}-y)
$$

and

$$
\frac{\partial J}{\partial b}
=============================

\frac{1}{m}\sum(\hat{y}-y)
$$

The parameters are updated using:

$$
\theta = \theta - \alpha\frac{\partial J}{\partial\theta}
$$

$$
b = b - \alpha\frac{\partial J}{\partial b}
$$

where `α` is the learning rate.

## Training

Example initialization:

```python
m, n = X_train.shape

theta = np.zeros((n, 1))
b = 0

learning_rate = 0.001
epochs = 5000
```

Training loop:

```python
for epoch in range(epochs):

    z = X_train @ theta + b

    y_pred = sigmoid(z)

    cost = compute_cost(y_train, y_pred)

    dw = gradient_w(X_train, y_pred, y_train)
    db = gradient_b(y_pred, y_train)

    theta = theta - learning_rate * dw
    b = b - learning_rate * db
```

## Data Preprocessing

### Customer ID

`customerID` was removed because it is an identifier and does not provide meaningful predictive information.

```python
df.drop('customerID', axis=1, inplace=True)
```

### Binary Encoding

Binary categorical variables were converted into numerical values.

For example:

```python
df['Partner'] = df['Partner'].map({
    'Yes': 1,
    'No': 0
})
```

Similarly, `Churn` was converted to:

```text
Yes → 1
No  → 0
```

### One-Hot Encoding

Multi-category features were converted using one-hot encoding.

Examples include:

* InternetService
* Contract
* PaymentMethod
* OnlineSecurity
* TechSupport
* StreamingTV
* StreamingMovies

### Numerical Scaling

The following numerical features were standardized:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Standardization was performed using the training data and then applied to the test data.

## Model Evaluation

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score

### Confusion Matrix

The obtained confusion matrix was approximately:

```text
[[242, 17],
 [ 64, 30]]
```

Interpretation:

|          | Predicted 0 | Predicted 1 |
| -------- | ----------: | ----------: |
| Actual 0 |         242 |          17 |
| Actual 1 |          64 |          30 |

Therefore:

* True Negatives = 242
* False Positives = 17
* False Negatives = 64
* True Positives = 30

### Accuracy

The model achieved approximately:

```text
Accuracy = 0.7705
```

or approximately:

```text
77.05%
```

However, accuracy alone is not sufficient for evaluating a churn prediction model.

The model identifies non-churn customers considerably better than churn customers, as shown by the relatively high number of false negatives.

## Important Observation

The model was trained and evaluated using a **95/5 train-test split**.

This means approximately:

```text
95% → Training data
5%  → Testing data
```

The small test set means the evaluation metrics can be sensitive to individual predictions.

A future improvement would be to use:

* 80/20 train-test split
* Stratified splitting
* K-fold cross-validation

This would provide a more reliable estimate of model performance.

## Visualizations

The project can include the following visualizations:

### Cost vs Epochs

Shows whether Gradient Descent is successfully minimizing the loss.

```python
plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost vs Epoch")
plt.show()
```

### Confusion Matrix

Visualizes:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC Curve

Shows the relationship between:

* True Positive Rate
* False Positive Rate

and allows calculation of ROC-AUC.

### Feature Coefficients

The learned values of `theta` can be used to understand how features influence the prediction.

Positive coefficients increase the tendency toward churn, while negative coefficients decrease it, assuming the features are encoded consistently.

## Why Implement Logistic Regression From Scratch?

The main objective of this project was not simply to achieve the highest possible prediction accuracy.

Implementing the algorithm from scratch provides a deeper understanding of:

* Linear models
* Sigmoid activation
* Probability estimation
* Binary cross-entropy
* Derivatives
* Gradient descent
* Parameter optimization
* Classification
* Model evaluation

Instead of calling:

```python
LogisticRegression().fit(X, y)
```

the core learning algorithm was implemented manually using NumPy.

## Future Improvements

Possible improvements include:

### 1. Better Train/Test Strategy

Use stratified 80/20 splitting or K-fold cross-validation.

### 2. Threshold Optimization

Instead of always using:

```python
threshold = 0.5
```

experiment with different thresholds to improve recall.

This is particularly useful because missing a customer who is likely to churn can be costly.

### 3. Regularization

Implement:

* L1 Regularization
* L2 Regularization

This can help reduce overfitting.

### 4. Hyperparameter Tuning

Experiment with:

* Learning rate
* Number of epochs
* Classification threshold
* Regularization strength

### 5. ROC-AUC

Add ROC-AUC to evaluate how well the model separates churners from non-churners.

### 6. Precision-Recall Curve

Because churn prediction can involve class imbalance, a Precision-Recall curve can provide additional insight.

### 7. Compare Against Scikit-learn

After implementing the algorithm from scratch, compare its results against Scikit-learn's Logistic Regression as a benchmark.

The goal is not to use Scikit-learn in the prediction pipeline, but to verify that the custom implementation behaves correctly.

### 8. Deployment

The trained model could eventually be deployed using:

* Streamlit
* Flask
* FastAPI

A simple application could allow a user to enter customer information and receive a predicted churn probability.

## Key Learning Outcomes

Through this project, I implemented a complete machine learning pipeline without relying on a pre-built Logistic Regression implementation.

The project demonstrates understanding of:

```text
Data preprocessing
       ↓
Feature engineering
       ↓
Mathematical modeling
       ↓
Sigmoid function
       ↓
Cost function
       ↓
Gradient computation
       ↓
Gradient descent
       ↓
Prediction
       ↓
Model evaluation
```

## Project Status

**Completed**

Core Logistic Regression model:

* [x] Data cleaning
* [x] Categorical encoding
* [x] One-hot encoding
* [x] Feature scaling
* [x] Train/test split
* [x] Logistic Regression from scratch
* [x] Sigmoid function
* [x] Binary cross-entropy
* [x] Gradient descent
* [x] Predictions
* [x] Confusion matrix
* [x] Accuracy evaluation

Potential next steps:

* [ ] Precision / Recall / F1 analysis
* [ ] ROC curve and AUC
* [ ] Precision-Recall curve
* [ ] Threshold optimization
* [ ] Regularization
* [ ] Cross-validation
* [ ] Model deployment

## Conclusion

This project demonstrates a complete implementation of Logistic Regression for customer churn prediction using NumPy.

The model achieved approximately **77.05% accuracy** on the test set. While the accuracy is reasonable, the confusion matrix shows that the model misses a significant number of actual churn customers.

This highlights an important lesson in machine learning: **a good classification model should not be judged by accuracy alone**. For churn prediction, recall, precision, F1 score, ROC-AUC, and the business cost of false negatives should also be considered.

```
```

The custom NumPy implementation achieved 78.42% accuracy, compared with 80.62% using Scikit-learn's Logistic Regression. The custom model achieved a churn precision of 0.67 and recall of 0.36, while Scikit-learn achieved 0.66 precision and 0.56 recall. The higher recall of the Scikit-learn model indicates that its optimization and regularization configuration was more effective at identifying customers who actually churned.