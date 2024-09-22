# Import necessary libraries
# -*- coding: utf-8 -*-
import pandas as pd
import lime
import lime.lime_tabular    
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_path = r"C:\Users\adeet\MLOPS_Group_10_Assignment_2\static\data\heart_disease_uci.csv"  # Adjust the path as needed
df = pd.read_csv(data_path)

# Print the columns
print(df.columns)

# Check the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Overview of the dataset
print("\nSummary statistics:")
print(df.describe())

print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Perform One-Hot Encoding on categorical columns
# 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca' are potential categorical columns
df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca', 'dataset'], drop_first=True)

# Feature and target separation
X = df_encoded.drop('num', axis=1)  # Drop 'num' (target) from the features
y = df_encoded['num']  # 'num' is the target variable

# Impute missing values using mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Preview the preprocessed data
print(X_train_scaled[:5])

# Define a dictionary of models and their parameter grids for hyperparameter tuning
models = {
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
}

# # To store the results of the best models
# best_models = []

# # Loop over the models and tune hyperparameters using GridSearchCV
# for model_name, model_info in models.items():
#     clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
#     clf.fit(X_train_scaled, y_train)
    
#     # Get the best model
#     best_model = clf.best_estimator_
#     best_score = clf.best_score_
    
#     # Predict on the test set
#     y_pred = best_model.predict(X_test_scaled)
    
#     # Evaluate the model on the test set
#     test_accuracy = accuracy_score(y_test, y_pred)
    
#     # Save the best model's results
#     best_models.append({
#         'Model': model_name,
#         'Best Params': clf.best_params_,
#         'Training Accuracy': best_score,
#         'Test Accuracy': test_accuracy
#     })
    
#     print(f"Model: {model_name}")
#     print(f"Best Params: {clf.best_params_}")
#     print(f"Training Accuracy: {best_score}")
#     print(f"Test Accuracy: {test_accuracy}")
#     print(f"Classification Report:\n {classification_report(y_test, y_pred)}")
#     print("-" * 60)

# # Convert the results into a DataFrame for comparison
# best_models_df = pd.DataFrame(best_models)
# print(best_models_df)

# To store the best model
best_model = None
best_params = None
best_accuracy = 0

# Loop over the models and tune hyperparameters using GridSearchCV
for model_name, model_info in models.items():
    clf = GridSearchCV(model_info['model'], model_info['params'], cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Get the best model
    if clf.best_score_ > best_accuracy:
        best_model = clf.best_estimator_
        best_params = clf.best_params_
        best_accuracy = clf.best_score_

    # Print model results
    print(f"Model: {model_name}")
    print(f"Best Params: {clf.best_params_}")
    print(f"Training Accuracy: {clf.best_score_}")
    print("-" * 60)

# Use the best model to make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Evaluate the best model on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy of Best Model: {test_accuracy}")
print(f"Classification Report:\n {classification_report(y_test, y_pred)}")

# Apply LIME for a single instance explanation
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_scaled, feature_names=df_encoded.columns, class_names=['No Disease', 'Disease'], discretize_continuous=True)

# Choose an instance to explain
instance = X_test_scaled[0]

# Generate explanation for a single instance
exp = explainer.explain_instance(instance, best_model.predict_proba, num_features=10)

# Option 1: If you're using a Jupyter notebook or an environment with IPython
exp.show_in_notebook(show_table=True)

# Option 2: Save explanation to an HTML file for non-notebook environments
exp.save_to_file('python lime_explanation.html')

