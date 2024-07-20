# app/model_building.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import joblib

def build_supervised_model(df, target_column, model_type='linear_regression', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    if model_type == 'linear_regression':
        performance = mean_squared_error(y_test, predictions)
    else:
        performance = accuracy_score(y_test, predictions)
    
    return model, performance

def build_unsupervised_model(df, model_type='kmeans', n_clusters=3, random_state=42):
    # Encode categorical variables
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    if model_type == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        raise ValueError("Unsupported model type")

    model.fit(df)
    labels = model.labels_
    
    return model, labels

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)
