"""
Data Preprocessing Module for SecureFed-NIDS
==============================================
Handles data cleaning, encoding, scaling, and feature selection.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from config import PREPROCESSING_CONFIG


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline for network intrusion detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or PREPROCESSING_CONFIG
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        method = self.config.get("handle_missing", "drop")

        if method == "drop":
            # Drop rows with any missing values
            initial_rows = len(df)
            df = df.dropna()
            print(f"Dropped {initial_rows - len(df)} rows with missing values")
        elif method == "fill":
            # Fill numeric columns with median, categorical with mode
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_vals = df[col].mode()
                    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
                    df[col] = df[col].fillna(fill_val)
        elif method == "interpolate":
            # Linear interpolation for numeric columns
            df = df.interpolate(method='linear', limit_direction='both')

        return df

    def encode_labels(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Encode target labels."""
        y = self.label_encoder.fit_transform(df[target_col])
        df = df.drop(columns=[target_col])

        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(y)}")

        return df, y

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        method = self.config.get("encoding", "label")

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns: {categorical_cols}")

        if method == "label":
            # Label encoding for each categorical column
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        elif method == "onehot":
            # One-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features using configured method."""
        method = self.config.get("normalization", "standard")

        if fit:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            elif method == "robust":
                self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def select_features(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> np.ndarray:
        """Select most important features."""
        method = self.config.get("feature_selection", "variance_threshold")

        if method == "variance_threshold":
            threshold = self.config.get("variance_threshold", 0.01)
            if fit:
                self.feature_selector = VarianceThreshold(threshold=threshold)
                X_selected = self.feature_selector.fit_transform(X)
                # Get selected feature indices
                selected_mask = self.feature_selector.get_support()
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1]) if selected_mask[i]]
            else:
                X_selected = self.feature_selector.transform(X)

        elif method == "correlation":
            # Remove highly correlated features
            if fit:
                df = pd.DataFrame(X)
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                self.feature_selector = to_drop
                X_selected = np.delete(X, to_drop, axis=1)
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1]) if i not in to_drop]
            else:
                X_selected = np.delete(X, self.feature_selector, axis=1)

        return X_selected

    def preprocess_full_pipeline(self, df: pd.DataFrame, target_col: str,
                                  test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline.

        Returns:
            Dictionary containing X_train, X_test, y_train, y_test, and metadata
        """
        print("\n" + "="*60)
        print("Starting Preprocessing Pipeline")
        print("="*60)

        # Step 1: Handle missing values
        print("\n[1/6] Handling missing values...")
        df = self.handle_missing_values(df)

        # Step 2: Encode categorical features
        print("\n[2/6] Encoding categorical features...")
        df = self.encode_categorical(df)

        # Step 3: Separate features and labels
        print("\n[3/6] Separating features and labels...")
        y_df, y = self.encode_labels(df, target_col)

        # Step 4: Split before feature selection & scaling to prevent train/test leakage
        print("\n[4/6] Splitting data...")
        min_count = int(np.min(np.bincount(y))) if len(y) > 0 else 0
        strat = y if min_count >= 2 else None
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            y_df.values, y,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )

        # Step 5: Feature selection (fit on training split only)
        print("\n[5/6] Selecting features...")
        X_train_sel = self.select_features(X_train_raw, y_train, fit=True)
        X_test_sel = self.select_features(X_test_raw, y_test, fit=False)

        # Step 6: Feature scaling (fit on training split only)
        print("\n[6/6] Scaling features...")
        X_train = self.scale_features(X_train_sel, fit=True)
        X_test = self.scale_features(X_test_sel, fit=False)

        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples")
        print("="*60 + "\n")

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "label_classes": self.label_encoder.classes_.tolist()
        }

    def transform_new_data(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        # Encode categorical
        X = self.encode_categorical(X)

        # Select features
        X = self.select_features(X, None, fit=False)

        # Scale
        X = self.scale_features(X, fit=False)

        return X


def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 256,
                       shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Simple data loader for batch training."""
    if shuffle:
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]

    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]


if __name__ == "__main__":
    # Example usage
    from config import DATASET_CONFIG

    preprocessor = DataPreprocessor()

    # Example with synthetic data for testing
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(20)])
    df['label'] = y

    result = preprocessor.preprocess_full_pipeline(df, 'label')
    print("Preprocessing test completed successfully!")