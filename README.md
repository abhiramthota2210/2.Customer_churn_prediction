# Telco Customer Churn Prediction

A customer churn prediction project using **Logistic Regression implemented from scratch with NumPy**.

The main goal of this project was to understand how Logistic Regression works internally rather than directly using a pre-built prediction algorithm.

## Dataset

The project uses the **Telco Customer Churn** dataset.

The target variable is:

- `Churn = 1` → Customer churned
- `Churn = 0` → Customer did not churn

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Scikit-learn was **not used for the prediction pipeline**. It was only used for preprocessing and evaluating/comparing the model.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed `customerID`
- Converted binary categorical features into numerical values
- Applied one-hot encoding to multi-category features
- Converted `TotalCharges` to numeric
- Standardized numerical features:
  - `tenure`
  - `MonthlyCharges`
  - `TotalCharges`
- Split the dataset into training and testing sets using a **95/5 split**

## Logistic Regression From Scratch

The Logistic Regression model was implemented using NumPy.

The model uses:

### Linear Model

```python
z = X @ theta + b

Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
Binary Cross-Entropy Cost

The Binary Cross-Entropy loss was implemented manually.

Gradient Descent

The model parameters were updated using:

theta = theta - learning_rate * dw
b = b - learning_rate * db

The model was trained for:

Learning rate: 0.001
Epochs: 5000
Model Evaluation

The model was evaluated using:

Accuracy
Confusion Matrix
Precision
Recall
F1 Score
Results

The from-scratch Logistic Regression model achieved:

Accuracy: 78.42%

Confusion Matrix:

[[969  66]
 [238 136]]

Classification results:

Class	Precision	Recall	F1-Score
0	0.80	0.94	0.86
1	0.67	0.36	0.47

The model performs better at identifying customers who do not churn than customers who do churn.

Comparison with Scikit-learn

As a reference, the results were compared with Scikit-learn's Logistic Regression.

From Scratch
Accuracy: 78.42%
Scikit-learn
Accuracy: 80.62%

Scikit-learn achieved higher recall and F1-score for the churn class.

Model	Accuracy	Churn Precision	Churn Recall	Churn F1
From Scratch	78.42%	0.67	0.36	0.47
Scikit-learn	80.62%	0.66	0.56	0.60
Key Learning

This project helped me understand the internal working of Logistic Regression, including:

Data preprocessing
Feature encoding
Feature scaling
Sigmoid function
Binary Cross-Entropy
Gradient calculation
Gradient Descent
Classification
Confusion Matrix
Precision, Recall and F1-score

The core Logistic Regression prediction algorithm was implemented from scratch using NumPy.
