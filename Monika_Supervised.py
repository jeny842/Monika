# Data manipulation and numerical computation
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning model and evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# For stats analysis
import statsmodels.api as sm
# Load dataset
data = pd.read_csv(r'C:\Users\Admin\Desktop\Monika_Social_Network_Ads.csv')

# View the first few rows
# print(data.head())
# print(data.sample(5))
# print(data.head())
# print(data.tail())
# print(data.shape)
# print(data.columns)
# print(data.dtypes)
# print(data.info())
# print(data.nunique())
# print(data.describe())

# # Check for missing values
# print(data.isnull().sum())

# # Drop unnecessary columns (e.g., User ID if present)
data = data.drop(['User ID'], axis=1)

# # Encode categorical data (e.g., Gender)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# # Distribution of ages by purchase status
sns.histplot(data=data, x='Age', hue='Purchased', multiple='stack')
plt.title('Age Distribution by Purchase')
plt.show()

# # Salary vs Age with Purchase status
sns.scatterplot(data=data, x='Age', y='EstimatedSalary', hue='Purchased')
plt.title('Estimated Salary vs Age by Purchase')
plt.show()

# # Define features and target
x = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

# # Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# # Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# # Fit Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(x_train_scaled, y_train)

# # Predict using the test set
# y_pred_lr = lr_model.predict(x_test_scaled)

# # Evaluation
# print('Logistic Regression Metrics:')
# print('Accuracy: {accuracy_score(y_test, y_pred_lr)*100:.2f}%')
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_lr))
# print('Classification Report:\n', classification_report(y_test, y_pred_lr))

# # Fit Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train_scaled, y_train)

# # Predict using the test set
# y_pred_rf = rf_model.predict(x_test_scaled)

# # Evaluation
# print('Random Forest Metrics:')
# print('Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%')
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_rf))
# print('Classification Report:\n', classification_report(y_test, y_pred_rf))

# # Cross-validation for Logistic Regression
cv_scores_lr = cross_val_score(lr_model, x_train_scaled, y_train, cv=5)
# print(f'Logistic Regression Cross-Validation Scores: {cv_scores_lr}')
# print(f'Average Cross-Validation Score: {cv_scores_lr.mean():.2f}')

# # Cross-validation for Random Forest
cv_scores_rf = cross_val_score(rf_model, x_train_scaled, y_train, cv=5)
# print(f'Random Forest Cross-Validation Scores: {cv_scores_rf}')
# print(f'Average Cross-Validation Score: {cv_scores_rf.mean():.2f}')

# # Hyperparameter tuning for Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5)
grid_search_lr.fit(X_train_scaled, y_train)

# print('Best parameters for Logistic Regression:', grid_search_lr.best_params_)

# # Hyperparameter tuning for Random Forest
param_grid_rf = {
     'n_estimators': [50, 100, 200],
     'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
 }
 grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5)
 grid_search_rf.fit(X_train_scaled, y_train)

# print('Best parameters for Random Forest:', grid_search_rf.best_params_)

# # Using the best model from GridSearchCV (Random Forest in this case)
best_rf_model = grid_search_rf.best_estimator_
best_rf_model.fit(X_train_scaled, y_train)
y_pred_best_rf = best_rf_model.predict(x_test_scaled)

# # Evaluation of the final model
# print('Best Random Forest Model Metrics:')
# print(f'Accuracy: {accuracy_score(y_test, y_pred_best_rf)*100:.2f}%')
# print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_best_rf))
# print('Classification Report:\n', classification_report(y_test, y_pred_best_rf))

# # Visualizing the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# # Feature Importance from Random Forest
feature_importances = best_rf_model.feature_importances_
features = x.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.show()
