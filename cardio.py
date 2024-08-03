# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
df = pd.read_csv('cardiovascular_risk_data.csv')  # Ensure the CSV file is in the same directory as your script

# Display the first few rows of the dataset
df.head()

# Data Preprocessing
# Handling missing values
df = df.dropna()

# Encoding categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Feature Engineering
df['age_binned'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80, 100], labels=False)
df['cholesterol_binned'] = pd.cut(df['cholesterol'], bins=[0, 200, 240, 280, 320, 360, 400, 500], labels=False)
df['bp_binned'] = pd.cut(df['blood_pressure'], bins=[0, 80, 90, 100, 110, 120, 130, 140, 150, 200], labels=False)

# Feature Selection
X = df.drop('cardiovascular_risk', axis=1)
y = df['cardiovascular_risk']

# Recursive Feature Elimination
model = LogisticRegression()
rfe = RFE(model, 10)
fit = rfe.fit(X, y)
X = X[X.columns[fit.support_]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training and Hyperparameter Tuning
# Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
rf_pred = rf_best_model.predict(X_test)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Neural Network
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.5))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=50, batch_size=10, validation_split=0.2)
nn_pred = nn_model.predict_classes(X_test)

# Model Evaluation
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f'Accuracy: {accuracy}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)
    print(f'ROC-AUC: {roc_auc}')
    
    # Visualize the Confusion Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

print("Logistic Regression Evaluation")
evaluate_model(y_test, log_pred)

print("Random Forest Evaluation")
evaluate_model(y_test, rf_pred)

print("Gradient Boosting Evaluation")
evaluate_model(y_test, gb_pred)

print("Neural Network Evaluation")
evaluate_model(y_test, nn_pred)
