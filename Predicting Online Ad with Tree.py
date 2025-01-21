import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Load the first 300,000 rows of the CSV file into a DataFrame
n_rows = 300000
df = pd.read_csv(r"C:\Users\johnp\Downloads\Predicting Online Ad with Tree\train\train.csv", nrows=n_rows)

# Quick look at the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(df.head(5))

# List all the column headers in the CSV
headers = df.columns.tolist()  # Use parentheses to call .tolist()
print("\nHeaders in the dataset:")
print(headers)

# Analyze the class distribution of the target variable ('click')
class_distribution = df['click'].value_counts()
print("\nClass distribution of the target variable ('click'):")
print(class_distribution)

# Calculate the positive class True Frequency (CTF)
total_samples = sum(class_distribution)
positive_ctf = class_distribution[1] / total_samples * 100
print(f"\nPositive Class True Frequency (CTF): {positive_ctf:.2f}%")

# Define target variable (Y) and features (X)
Y = df['click'].values  # Target variable we're trying to predict
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values  # Drop irrelevant columns
print(f"\nShape of feature matrix (X): {X.shape}")

# Split data into training and testing sets
# 90% for training, 10% for testing (chronological order to avoid using future data to predict past data)
n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

# Transform categorical data into numerical data using One-Hot Encoding
# Scikit-learn requires numerical inputs for training
enc = OneHotEncoder(handle_unknown='ignore')  # Handle unknown categories gracefully
X_train_enc = enc.fit_transform(X_train)  # Fit and transform training data
print("\nExample of a transformed training sample (sparse vector):")
print(X_train_enc[0])

# Transform the test data using the same encoder
X_test_enc = enc.transform(X_test)

# Use GridSearchCV to perform cross-validation and find the best hyperparameters for the decision tree
parameters = {'max_depth': [3, 10, None]}  # Hyperparameters to tune
decision_tree = DecisionTreeClassifier(criterion='gini', min_samples_split=30)  # Initialize DecisionTreeClassifier

grid_search = GridSearchCV(
    decision_tree,
    parameters,
    n_jobs=-1,  # Utilize all CPU cores for faster computation
    cv=3,       # 3-fold cross-validation
    scoring='roc_auc'  # Use ROC-AUC as the evaluation metric
)

# Fit the grid search on the training data
grid_search.fit(X_train_enc, Y_train)

# Print the best hyperparameters and ROC-AUC score on the test set
print("\nBest parameters found using GridSearchCV:")
print(grid_search.best_params_)
decision_tree_best = grid_search.best_estimator_  # Retrieve the best estimator
pos_prob = decision_tree_best.predict_proba(X_test_enc)[:, 1]  # Predicted probabilities for the positive class

# Calculate the ROC-AUC score on the test set
roc_auc = roc_auc_score(Y_test, pos_prob)
print(f"\nThe ROC-AUC on the testing set is: {roc_auc:.3f}")

# Simulate a random selection baseline for comparison
pos_prob_random = np.zeros(len(Y_test))  # Initialize probabilities with all zeros
click_index = np.random.choice(
    len(Y_test),
    int(len(Y_test) * 51211.0 / 300000),  # Randomly select indices based on class distribution
    replace=False
)
pos_prob_random[click_index] = 1  # Assign positive probabilities to randomly selected indices

# Calculate the ROC-AUC for the random selection baseline
roc_auc_random = roc_auc_score(Y_test, pos_prob_random)
print(f"The ROC-AUC using random selection is: {roc_auc_random:.3f}")

#This is likely to cause overfitting as optimal points only work well for the training samples. 
#We use ensembling to correct this, with random forest ensemble
#Best practice is to 1. Encode Categorical Features
#                    2. Scale Numerical Features (So Bigger numbers don't dominate, use Normalization and Standardization)
#Random Forest uses the concept of Tree Bagging, reducing high variance that occurs in a single tree
from sklearn.ensemble import RandomForestClassifier

# Random Forest Classifier with verbose output
random_forest = RandomForestClassifier(
    n_estimators=100,          # Number of trees in the forest
    criterion='gini',          # Use Gini impurity for splitting
    min_samples_split=30,      # Minimum samples required to split a node
    n_jobs=-1,                 # Use all available CPU cores
    verbose=1                  # Enable verbose output
)

# Perform GridSearchCV for Random Forest with verbose output
grid_search = GridSearchCV(
    random_forest,
    parameters,
    n_jobs=-1,
    cv=3,                      # 3-fold cross-validation
    scoring='roc_auc',         # Evaluate using ROC-AUC metric
    verbose=2                  # Enable verbose output for GridSearchCV
)

# Fit the grid search on the training data
grid_search.fit(X_train_enc, Y_train)

# Print the best parameters found during the grid search
print("\nBest parameters found using GridSearchCV for Random Forest:")
print(grid_search.best_params_)

# Evaluate the best Random Forest model
random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print(f"\nThe ROC-AUC on the testing set using Random Forest is: {roc_auc_score(Y_test, pos_prob):.3f}")
