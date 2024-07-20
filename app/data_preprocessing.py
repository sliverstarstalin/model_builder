import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import joblib

def load_data(file_path):
    if file_path.endswith('.csv'):
        try:
            if 'boston' in file_path.lower():
                columns = [
                    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
                    "PTRATIO", "B", "LSTAT", "MEDV"
                ]
                return pd.read_csv(file_path, delim_whitespace=True, names=columns)
            elif 'iris' in file_path.lower():
                columns = [
                    "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
                ]
                return pd.read_csv(file_path, names=columns)
            elif 'wine' in file_path.lower():
                columns = [
                    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", 
                    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
                ]
                return pd.read_csv(file_path, names=columns)
            else:
                return pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='latin1')
    elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

def handle_missing_values(df, strategy='mean'):
    """Handle missing values in the dataframe based on the given strategy."""
    if strategy == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif strategy == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    elif callable(strategy):
        return df.apply(strategy)
    else:
        raise ValueError("Unsupported strategy")

def detect_outliers(df, method='z_score', threshold=3):
    """Detect outliers in the dataframe based on the given method."""
    numerical_df = df.select_dtypes(include=[np.number])
    if method == 'z_score':
        return numerical_df[(numerical_df - numerical_df.mean()).abs() > threshold * numerical_df.std()]
    elif method == 'iqr':
        Q1 = numerical_df.quantile(0.25)
        Q3 = numerical_df.quantile(0.75)
        IQR = Q3 - Q1
        return numerical_df[((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).any(axis=1)]
    else:
        raise ValueError("Unsupported method")

def remove_outliers(df, method='z_score', threshold=3):
    """Remove outliers from the dataframe."""
    outliers = detect_outliers(df, method, threshold)
    return df.drop(outliers.index)

def normalize_data(df):
    """Normalize the data in the dataframe."""
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def standardize_data(df):
    """Standardize the data in the dataframe."""
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def encode_categorical(df, method='onehot'):
    """Encode categorical columns in the dataframe."""
    if method == 'onehot':
        return pd.get_dummies(df)
    elif method == 'label':
        encoder = LabelEncoder()
        return df.apply(encoder.fit_transform)
    else:
        raise ValueError("Unsupported method")

def convert_data_types(df, columns, data_type):
    """Convert data types of specified columns."""
    for column in columns:
        df[column] = df[column].astype(data_type)
    return df

def handle_duplicates(df, method='drop'):
    """Handle duplicates in the dataframe."""
    if method == 'drop':
        return df.drop_duplicates()
    elif method == 'keep_first':
        return df.drop_duplicates(keep='first')
    elif method == 'keep_last':
        return df.drop_duplicates(keep='last')
    else:
        raise ValueError("Unsupported method")

def log_and_report(df, report_file='data_report.txt'):
    """Log and report the dataframe statistics and missing values."""
    with open(report_file, 'w') as f:
        f.write(df.describe().to_string())
        f.write('\n\nMissing Values:\n')
        f.write(df.isnull().sum().to_string())
    return df

def save_preprocessing_pipeline(pipeline, file_path='preprocessing_pipeline.pkl'):
    """Save the preprocessing pipeline to a file."""
    joblib.dump(pipeline, file_path)

def load_preprocessing_pipeline(file_path='preprocessing_pipeline.pkl'):
    """Load the preprocessing pipeline from a file."""
    return joblib.load(file_path)

def compute_data_metrics(df):
    """Compute and return various metrics for the dataframe."""
    metrics = {}

    # Missing values
    missing_values = df.isnull().sum()
    metrics['missing_values'] = missing_values.to_dict()
    metrics['missing_percentage'] = (missing_values / len(df) * 100).to_dict()

    # Outliers
    metrics['outliers'] = detect_outliers(df).shape[0]

    # Data distribution
    metrics['data_distribution'] = df.describe().to_dict()

    # Categorical data
    categorical_metrics = {}
    for column in df.select_dtypes(include=['object']).columns:
        categorical_metrics[column] = df[column].nunique()
    metrics['categorical_data'] = categorical_metrics

    # Data types
    metrics['data_types'] = df.dtypes.to_dict()

    return metrics
