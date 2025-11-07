"""
Data Preprocessing Module for Fraud Detection
Handles feature engineering, scaling, and data preparation
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """Preprocessor for credit card fraud detection data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features from raw transaction data
        
        Args:
            df: Input dataframe with transaction data
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Extract time-based features if Time column exists
        if 'Time' in df.columns:
            # Convert seconds to hours
            df['hour'] = (df['Time'] / 3600) % 24
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            # Time period features
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Amount-based features
        if 'Amount' in df.columns:
            # Log transform for amount (handle zeros)
            df['log_amount'] = np.log1p(df['Amount'])
            
            # Amount categories
            df['amount_category'] = pd.cut(
                df['Amount'],
                bins=[-np.inf, 10, 100, 500, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(int)
            
            # Statistical features
            df['amount_squared'] = df['Amount'] ** 2
            df['amount_sqrt'] = np.sqrt(df['Amount'])
        
        # V-feature interactions (if V1, V2, etc. exist)
        v_cols = [col for col in df.columns if col.startswith('V')]
        if len(v_cols) >= 2:
            # Create interaction features for top correlated V features
            df['v_interaction_1'] = df.get('V1', 0) * df.get('V2', 0)
            df['v_interaction_2'] = df.get('V3', 0) * df.get('V4', 0)
            
            # Statistical aggregations
            df['v_mean'] = df[v_cols].mean(axis=1)
            df['v_std'] = df[v_cols].std(axis=1)
            df['v_max'] = df[v_cols].max(axis=1)
            df['v_min'] = df[v_cols].min(axis=1)
        
        logger.info(f"Engineered features. New shape: {df.shape}")
        return df
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Class',
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for modeling
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (features, target)
        """
        df = self.engineer_features(df)
        
        # Separate features and target
        target = None
        if target_col in df.columns:
            target = df[target_col]
            df = df.drop(target_col, axis=1)
        
        # Drop Time column if exists
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        # Handle missing values
        df = df.fillna(df.median())
        
        # Store feature names
        if fit:
            self.feature_names = df.columns.tolist()
        
        # Ensure consistent columns
        if self.feature_names is not None:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0
            # Keep only known columns in order
            df = df[self.feature_names]
        
        logger.info(f"Prepared {len(df.columns)} features")
        return df, target
    
    def scale_features(
        self, 
        X: pd.DataFrame, 
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature dataframe
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names if self.feature_names else []


def load_and_preprocess_data(
    filepath: str,
    preprocessor: Optional[FraudPreprocessor] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], FraudPreprocessor]:
    """
    Load and preprocess data from CSV
    
    Args:
        filepath: Path to CSV file
        preprocessor: Existing preprocessor (optional)
        fit: Whether to fit preprocessor
        
    Returns:
        Tuple of (features, target, preprocessor)
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    if preprocessor is None:
        preprocessor = FraudPreprocessor()
    
    X, y = preprocessor.prepare_features(df, fit=fit)
    
    return X, y, preprocessor
